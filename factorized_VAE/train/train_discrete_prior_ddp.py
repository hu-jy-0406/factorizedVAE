import os
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid, save_image
import torch.multiprocessing as mp
from tqdm import tqdm
import wandb
from datetime import datetime
from tokenizer.tokenizer_image.xqgan_model import VQ_models
from PIL import Image
from factorized_VAE.my_models.discrete_prior import DiscretePrior
from factorized_VAE.my_models.lfq import LFQ_model
from factorized_VAE.utils import process_image
import numpy as np

# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_TIMEOUT=1800
# export TORCH_NCCL_TRACE_BUFFER_SIZE=10485760
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_TIMEOUT"] = "300"

#export PYTHONPATH=/home/renderex/causal_groups/jinyuan.hu/factorizedVAE:$PYTHONPATH
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"  # Set visible GPUs for DDP
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12348'  # Set master port for DDP
#os.environ["WANDB_MODE"] = "disabled"

def sample(model, vq_model, num_samples=10):
    """
    Generate samples from the model.
    """
    model.eval()
    samples = []

    batch_size = 1
    num_batches = math.ceil(num_samples / batch_size)
     

    for j in range(num_batches):
        batch_size = min(batch_size, num_samples - len(samples))

        tokens = model.generate(num_samples=batch_size)

        for i in range(tokens.shape[0]):
            tokens_i = tokens[i].reshape(8, 8)
            tokens_i = torch.tensor(tokens_i).unsqueeze(0)  # shape (1, 8, 8)
            latents_i = vq_model.quantize.codebook_lookup(tokens_i).squeeze(0)  # shape (1, 8, 8, codebook_embed_dim)
            img_i = vq_model.decode(latents_i).to('cpu')  # shape (1, 3, 256, 256)
            img_i = process_image(img_i)  # Convert to PIL Image
            samples.append(img_i)
    
    print(f"Generated {len(samples)} samples")

    # show all the sampled 32*32 PIL images in one image with 2x5 grid
    if len(samples) > 0:
        grid_img = Image.new('RGB', (32 * 5, 32 * 4))
        for i, img in enumerate(samples):
            grid_img.paste(img, (i % 5 * 32, i // 5 * 32))
        #grid_img.show()  # Display the grid of images
        grid_img.save("factorized_VAE/sampled_images2.png")  # Save the grid image
    else:
        print("No samples generated.")
    return samples


def main(rank, world_size):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #---------- DDP setup ----------#
    dist.init_process_group(
        backend="nccl",  # 指定后端通讯方式为nccl
        # init_method='tcp://localhost:23456',
        init_method='env://',  # 使用环境变量初始化
        rank=rank,  # rank是指当前进程的编号
        world_size=world_size  # worla_size是指总共的进程数
    )
    torch.cuda.set_device(rank)  # 设置当前进程使用的GPU
    print(f"Rank: {rank}, Device: {device}")

    #---------- Load Dataset ----------#
    path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-VAE-latent/CIFAR10-VAE-discrete-indices.pt"
    data = torch.load(path)
    dataset = TensorDataset(data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    batch_size = 64  # Set batch size for DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    #---------- Load Test Dataset ----------#
    # batch_size = 3
    # test_dataset = [dataset[1], dataset[2], dataset[3], dataset[4], dataset[5], dataset[6]]
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    # loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=0, pin_memory=True)
    
    # ------- Load LFQ-VAE Model ------- #
    #lfq_model = LFQ_model("factorized_VAE/ckpts/vae/kl16.ckpt").to(device).eval()
    lfq_model = LFQ_model().to(device).eval()
    resume_path = "/home/renderex/causal_groups/jinyuan.hu/ckpts/lfq_vae/lfq_vae_epoch10.pth"
    ckpt = torch.load(resume_path, map_location=device)
    lfq_model.load_state_dict(ckpt["model"])
    
    #---------- Initialize DiscretePrior ----------#
    vocab_size = 65536  # set this to match your codebook size
    seq_len = 256
    model = DiscretePrior(vocab_size, seq_len=seq_len, d_model=512, nhead=8, num_layers=8).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    #--------Hyperparameters----------#
    lr = 2e-4
    epochs = 8000
    vis_every = 100
    vis_num = 6
    save_every = 50  # Save checkpoint every 500 epochs
    
    #---------- Optimizer and Loss ----------#
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    #---------- Load ckpt ----------#
    resume_path = "/home/renderex/causal_groups/jinyuan.hu/ckpts/discrete_prior/DiscretePrior_CIFAR_epoch4000.pth"
    if resume_path and os.path.isfile(resume_path):
        print("Loading resume checkpoint...")
        ckpt = torch.load(resume_path, map_location=device)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        print(f"Resumed from epoch {start_epoch} at global step {global_step}.")
    else:
        print("No resume checkpoint found, starting from scratch.")
        start_epoch = 0
        global_step = 0
        
    #---------- Logging ----------#
    if rank == 0:  # Only initialize wandb in the main process
        wandb.login(key="3761ca3994faa9fea7e291585ce72a0ed49562a0")
        logger = wandb.init(project="DiscretePrior",
                            name=str(datetime.now().strftime('%Y.%m.%d-%H.%M.%S'))+f"rank-{rank}")
        #log total parameters
        logger.config.update({
            "total_params": total_params,
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "epochs": epochs,
        })

    #---------- Training Loop ----------#
    model.train()
    for epoch in range(start_epoch, epochs):
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        total_loss = 0
        for batch in pbar:
            
            batch = batch[0].to(device)# batch shape: (B, H, W)
            batch = batch.view(batch.size(0), -1)  # Flatten to (B, L)
            
            # Forward pass => logits (B, L, vocab_size)
            logits = model(batch)

            # We want to predict each token i from the previous tokens => cross-entropy
            # logits => (B, L, vocab_size); batch => (B, L)
            loss = criterion(logits.reshape(-1, vocab_size), batch.view(-1))
            
            global_step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)
            pbar.set_postfix(loss=loss.item())
            if rank == 0:
                logger.log({"batch_loss": loss.item()}, step=global_step)
        
        if rank == 0:
            logger.log({"epoch": epoch + 1}, step=global_step)

        # Visualize
        if rank == 0 and (epoch + 1) % vis_every == 0:
            print(f"Visualizing results at epoch {epoch + 1}...")
            model.eval()
            with torch.no_grad():
                
                # --------------- Visualize Logits --------------- #
                gen_indices = logits[:vis_num].argmax(dim=-1)  # shape (B, L)
                gen_latents = lfq_model.quantizer.indices_to_latents(gen_indices)  # shape (B, 256, 16)
                gen_latents = gen_latents.reshape(gen_latents.shape[0], 16, 16, 16)
                gen_imgs = lfq_model.decode_quantized_latent(gen_latents)  # shape (B, 3, 32, 32)
                gen_grid = make_grid(gen_imgs, nrow=vis_num, normalize=True, value_range=(-1, 1))
                
                gt_indices = batch[:vis_num].reshape(vis_num, 16, 16)
                gt_latents = lfq_model.quantizer.indices_to_latents(gt_indices)
                gt_imgs = lfq_model.decode_quantized_latent(gt_latents)  # shape (B, 3, 256, 256)
                gt_grid = make_grid(gt_imgs, nrow=vis_num, normalize=True, value_range=(-1, 1))
                combined_grid_fit = torch.cat((gt_grid, gen_grid), dim=1)
                # Convert to NumPy format
                combined_image_fit = combined_grid_fit.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
                combined_image_fit = (combined_image_fit * 255).astype(np.uint8)  # Convert
                logger.log({"Fitting Results": wandb.Image(combined_image_fit)}, step=global_step)
                # Save the combined image
                save_path = f"factorized_VAE/images/discrete_prior_fitting_res.png"
                save_image(combined_grid_fit, save_path, normalize=False) 
                
                # --------------- Visualize Generated Samples --------------- #
                gen_indices = model.module.generate(num_samples=vis_num)
                gen_latents = lfq_model.quantizer.indices_to_latents(gen_indices)  # shape (B, 256, 16)
                gen_latents = gen_latents.reshape(gen_latents.shape[0], 16, 16, 16)
                gen_imgs = lfq_model.decode_quantized_latent(gen_latents)  # shape (B, 3, 32, 32)
                gen_grid = make_grid(gen_imgs, nrow=vis_num, normalize=True, value_range=(-1, 1))
                
                # gt_indices = batch[:vis_num].reshape(vis_num, 16, 16)
                # gt_latents = lfq_model.quantizer.indices_to_latents(gt_indices)  # shape (B, 256, 16)
                # #gt_latents = gt_latents.permute(0, 3, 1, 2)
                # gt_imgs = lfq_model.decode_quantized_latent(gt_latents)  # shape (B, 3, 256, 256)
                # gt_grid = make_grid(gt_imgs, nrow=vis_num, normalize=True, value_range=(-1, 1))
                #combined_grid = torch.cat((gt_grid, gen_grid), dim=1)
                combined_grid = gen_grid  # Only visualize generated samples
                
                # Convert to NumPy format
                combined_image = combined_grid.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
                combined_image = (combined_image * 255).astype(np.uint8)  # Convert
                logger.log({"Generation Results": wandb.Image(combined_image)}, step=global_step)
                # Save the combined image
                save_path = f"factorized_VAE/images/discrete_prior_generated_samples.png"
                save_image(combined_grid, save_path, normalize=False)
                
            model.train()
        
        
        # Save checkpoint
        if rank == 0 and ((epoch+1) % save_every == 0 or (epoch+1) == epochs):
            ckpt_state = {
                "epoch": epoch+1,
                "global_step": global_step+1,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            ckpt_path = f"/home/renderex/causal_groups/jinyuan.hu/ckpts/discrete_prior/DiscretePrior_CIFAR_epoch{epoch+1}.pth"
            torch.save(ckpt_state, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
        #torch.save(model.state_dict(), f"factorized_VAE/token_transformer_decoder_epoch{epoch+1}.pth")
        dist.barrier()
        
    # ckpt = torch.load(resume_path, map_location=device)
    # model.load_state_dict(ckpt["model"])
    dist.barrier()  # Ensure all processes reach this point before proceeding
    dist.destroy_process_group()  # Destroy the process group after training
    print("Done!")

def run_ddp():
    world_size = torch.cuda.device_count()
    print(f"Running DDP with {world_size} GPUs")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    run_ddp()
    #单GPU：
    #main(0, 1)
    
    

