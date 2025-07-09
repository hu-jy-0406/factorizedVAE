import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from factorized_VAE.my_models.lfq import LFQ_model
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import os
import wandb
import numpy as np
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12347'
#os.environ["WANDB_MODE"] = "disabled"

#export PYTHONPATH=/home/renderex/causal_groups/jinyuan.hu/factorizedVAE:$PYTHONPATH

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
    batch_size = 20
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder("/home/renderex/causal_groups/jinyuan.hu/CIFAR10", transform=transform)
    print(f"Dataset size: {len(dataset)}")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    #---------- Load Test Dataset ----------#
    # test_dataset = [dataset[1], dataset[3], dataset[4]]
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=3, sampler=test_sampler, num_workers=0, pin_memory=True)
    
    #---------- Initialize LFQ -----------#
    model = LFQ_model("factorized_VAE/ckpts/vae/kl16.ckpt").to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    #--------Hyperparameters----------#
    lr = 2e-4
    epochs = 10
    save_every = 2
    vis_every = 1
    vis_num = 4
    
    #---------- Optimizer ----------#
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    #---------- Load ckpt ----------#
    resume_path = "factorized_VAE/ckpts/lfq_vae/lfq_vae_epoch1.pth"
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
        
    if rank == 0:  # Only initialize wandb in the main process
        wandb.login(key="3761ca3994faa9fea7e291585ce72a0ed49562a0")
        logger = wandb.init(project="LFQ-Decoder",
                            name=str(datetime.now().strftime('%Y.%m.%d-%H.%M.%S'))+f"rank-{rank}")
        #log total parameters
        logger.config.update({
            "total_params": total_params,
            "vocab_size": 2 ** model.module.embedding_dim,
            "batch_size": batch_size,
            "epochs": epochs
        })
        
    #---------- Training Loop ----------#
    model.train()
    for epoch in range(start_epoch, epochs):
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", disable=(rank != 0))
        
        total_loss = 0.0
        for batch in pbar:
            batch = batch[0].to(device)#batch shape = (batch_size, 3, 256, 256)
            recon_loss, ploss, _, x_recon = model(batch)
            loss = recon_loss + ploss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
            pbar.set_postfix(loss=loss.item())
            global_step += 1
            if rank == 0:
                logger.log({"batch_total_loss": loss.item()}, step=global_step)
                logger.log({"batch_recon_loss": recon_loss.item()}, step=global_step)
                logger.log({"batch_ploss": ploss.item()}, step=global_step)
                logger.log({"epoch": epoch + 1}, step=global_step)
                
        # Visualize reconstructions every vis_every epochs
        if rank == 0 and (epoch + 1) % vis_every == 0:
            #save_image(pred_dec[0], "factorized_VAE/ContinousPrior/pred_dec.png", normalize=True, value_range=(-1, 1))
            gen_grid = make_grid(x_recon[:vis_num], nrow=vis_num, normalize=True, value_range=(-1, 1))
            gt_grid = make_grid(batch[:vis_num], nrow=vis_num, normalize=True, value_range=(-1, 1))
            combined_grid = torch.cat((gt_grid, gen_grid), dim=1)
            # Convert to NumPy format
            combined_image = combined_grid.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
            combined_image = (combined_image * 255).astype(np.uint8)  # Convert
            logger.log({"Fitting Results": wandb.Image(combined_image)}, step=global_step)
            # Save the combined image
            save_path = f"factorized_VAE/images/lfq_decoder_recon.png"
            save_image(combined_grid, save_path, normalize=False)
            
        # Save checkpoint
        if rank == 0 and ((epoch+1) % save_every == 0 or (epoch+1) == epochs):
            ckpt_state = {
                "epoch": epoch+1,
                "global_step": global_step+1,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            ckpt_path = f"factorized_VAE/ckpts/lfq_vae/lfq_vae_epoch{epoch+1}.pth"
            torch.save(ckpt_state, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
            
    dist.barrier()  # Ensure all processes reach this point before exiting
    dist.destroy_process_group()  # Clean up the process group
    print(f"Rank {rank} finished training.")
    
def run_ddp():
    world_size = torch.cuda.device_count()
    print(f"Running DDP with {world_size} GPUs")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    
if __name__ == "__main__":
    run_ddp()