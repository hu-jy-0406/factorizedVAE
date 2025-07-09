import os
import torch
import math
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision.utils import make_grid, save_image
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from tqdm import tqdm
import wandb

from imagefolder_models.vae import AutoencoderKL

from factorized_VAE.my_models.discrete_prior import DiscretePrior
from factorized_VAE.my_models.continous_prior import ContinousPrior
from factorized_VAE.my_models.lfq import LFQ_quantizer
from factorized_VAE.utils import process_image

from PIL import Image
from evaluator import Evaluator

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta



# export MASTER_ADDR=localhost
# export MASTER_PORT=29500
# export WORLD_SIZE=1
# export RANK=0
# export LOCAL_RANK=0
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# export PYTHONPATH=/home/renderex/causal_groups/jinyuan.hu/factorizedVAE:$PYTHONPATH
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,6" # Using one GPU for debugging
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

os.environ["WANDB_MODE"] = "disabled"

# def setup(rank, world_size):
#     """
#     Initialize the distributed environment.
#     """
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     dist.init_process_group(backend='nccl', init_method='env://', 
#                             world_size=world_size, rank=rank,
#                             timeout=timedelta(seconds=10))

# def cleanup():
#     """
#     Clean up the distributed environment.
#     """
#     dist.destroy_process_group()

# class TokenDataset(Dataset):
#     def __init__(self, npy_path):
#         self.tokens = np.load(npy_path)  # shape (N, 8, 8)
#         self.tokens = self.tokens.reshape(self.tokens.shape[0], -1)  # shape (N, 64)

#     def __len__(self):
#         return self.tokens.shape[0]

#     def __getitem__(self, idx):
#         return torch.LongTensor(self.tokens[idx])
    
# class LatentDataset(Dataset):
#     def __init__(self, npy_path):
#         self.latents = np.load(npy_path)  # shape (N, 32, 8, 8)
#         self.latents = torch.FloatTensor(self.latents).permute(0, 2, 3, 1)  # shape (N, 8, 8, 32)
#         #reshape to (N, 64, 32)
#         N, H, W, C = self.latents.shape
#         self.latents = self.latents.reshape(N, H * W, C)  # shape (N, 64, 32)
#     def __len__(self):
#         return self.latents.shape[0]
#     def __getitem__(self, idx):
#         return self.latents[idx]



def sample(model, vq_model, discrete_prior, num_samples=20):
    """
    Generate samples from the model.
    """
    model.eval()
    samples = []

    batch_size = 20
    num_batches = math.ceil(num_samples / batch_size)
    

    for j in range(num_batches):
        batch_size = min(batch_size, num_samples - len(samples))

        tokens, latents = model.generate(discrete_prior, num_samples=batch_size)
        # tokens shape: (batch_size, seq_len=64)
        # latents shape: (batch_size, seq_len=64, z_dim=32)

        #-----check whether the tokens are correct-----
        #get latents from tokens
        # tokens = tokens.reshape(tokens.shape[0], 8, 8)  # Reshape to (batch_size, 8, 8)
        # latents = vq_model.quantize.codebook_lookup(tokens)  # shape (batch_size, 8, 8, z_dim)
        # latents = latents.permute(0, 2, 3, 1)  # Change to (batch_size, 8, 8, z_dim)
        # latents = latents.reshape(latents.shape[0], 64, -1)  # Reshape to (batch_size, 64, z_dim)
        #---------------------------------------------

        for i in range(latents.shape[0]):
            B, L, C = latents.shape
            latents_i = latents[i].reshape(8, 8, C)
            latents_i = torch.tensor(latents_i).unsqueeze(0).permute(0,3,1,2) # shape (1, C, 8, 8)

            #print(f"latents_i shape: {latents_i.shape}, tokens shape: {tokens[i].shape}")
            
            #latents_i,_,_,_,_ = vq_model.quantize(latents_i, ret_usages=True, dropout=None)# Quantize the latents

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
        grid_img.save("factorized_VAE/samples.png")  # Save the grid image
    else:
        grid_img = None
        print("No samples generated.")
    #show grid image in wandb

    return grid_img

    # generated_tokens = model.generate().cpu()  # shape (1, 64)
    # print("shape of generated tokens:", generated_tokens.shape)
    # print("Generated tokens:", generated_tokens)
    # #generated_tokens = model.autoregressive_inference(seq_len=64, start_token_id=0, gt=x_reshaped)
    # gen_tokens = np.array(generated_tokens).reshape(8, 8)
    # #change 8*8 array to 1*8*8 tensor
    # gen_tokens = torch.tensor(gen_tokens).unsqueeze(0)  # shape (1, 8, 8)

    # gen_latents = vq_model.quantize.codebook_lookup(gen_tokens)  # shape (1, 8, 8, codebook_embed_dim)
    # gen_latents = gen_latents.squeeze(0)  # shape (8, 8, codebook_embed_dim)
    # gen_img = vq_model.decode(gen_latents)  # shape (1, 3, 256, 256)
    # save_image(gen_img, "factorized_VAE/generated_image.png")
    

def load_discrete_backbone(continous_prior_model, discrete_prior_ckpt_path):
    """
    Load the TransformerDecoder weights from a DiscretePrior checkpoint into a ContinousPrior model.

    Args:
        continous_prior_model: The ContinousPrior model to initialize.
        discrete_prior_ckpt_path: Path to the DiscretePrior checkpoint.
    """
    # Load the DiscretePrior checkpoint
    ckpt = torch.load(discrete_prior_ckpt_path, map_location="cpu")
    discrete_prior_state_dict = ckpt["model"]

    print(f"Loading TransformerDecoder weights from {discrete_prior_ckpt_path} into ContinousPrior model...")

    # Extract the TransformerDecoder weights from the DiscretePrior checkpoint
    transformer_decoder_keys = [
        key for key in discrete_prior_state_dict.keys() if key.startswith("decoder.")
    ]

    # Map the keys to the ContinousPrior model
    continous_prior_state_dict = continous_prior_model.state_dict()
    for key in transformer_decoder_keys:
        # Replace "decoder." with "decoder." to match the ContinousPrior model's key structure
        #new_key = key.replace("decoder.", "decoder.")
        if key in continous_prior_state_dict:
            continous_prior_state_dict[key] = discrete_prior_state_dict[key]
        else:
            print(f"Warning: Key {key} not found in ContinousPrior model.")

    # Load the updated state dict into the ContinousPrior model
    continous_prior_model.load_state_dict(continous_prior_state_dict, strict=False)
    print("TransformerDecoder weights successfully loaded.")

def main(rank, world_size):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #---------- DDP setup ----------#
    dist.init_process_group(backend='nccl', init_method='env://', 
                            world_size=world_size, rank=rank,
                            timeout=timedelta(seconds=10))
    torch.cuda.set_device(rank)
    print(f"Rank: {rank}, Device: {device}")
    
    #---------- Load Dataset ----------#
    batch_size = 3
    path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-VAE-latent/CIFAR10-VAE-latents.pt"
    latents = torch.load(path)
    print(f"Loaded latents shape: {latents.shape}")
    dataset = TensorDataset(latents)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    print(f"Loaded latents dataset shape: {latents.shape}")
    
    #---------- Load Test Dataset ----------#
    test_dataset = [dataset[1], dataset[2], dataset[3], dataset[4], dataset[5], dataset[6]]
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(test_dataset, batch_size=3, sampler=test_sampler, num_workers=0, pin_memory=True)
       
    #----------Load DiscretePrior----------#
    vocab_size = 65536  # set this to match your codebook size
    seq_len = 256
    # discrete_prior = DiscretePrior(vocab_size=vocab_size, seq_len=seq_len, d_model=512, nhead=8, num_layers=8).to(device)
    # discrete_resume_path = "factorized_VAE/DiscretePrior_val_epoch2000.pth"
    # ckpt = torch.load(discrete_resume_path, map_location=device)
    # discrete_prior.load_state_dict(ckpt["model"])
    # discrete_prior = DDP(discrete_prior, device_ids=[rank])
    
    #---------- Load VAE ----------#
    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path="/home/renderex/causal_groups/jinyuan.hu/mar/pretrained_models/vae/kl16.ckpt").cuda().eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    #---------- Load Quantizer ----------#
    quantizer = LFQ_quantizer(embedding_dim=16).cuda().eval()  # Assuming embedding_dim is 16, adjust as needed
    for param in quantizer.parameters():
        param.requires_grad = False
    
    #-----------Initialize ContinousPrior-----------#
    model = ContinousPrior(quantizer=quantizer, vae=vae, z_dim=16, seq_len=seq_len, d_model=512, nhead=8, num_layers=8).to(device)
    model = DDP(model, device_ids=[rank])
    
    #----------- Hyperparameters -----------#
    lr = 1e-4
    epochs = 100
    vis_every = 1
    vis_num = 3
    save_every = 500
    
    # ---------- Optimizer ----------#
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    # ---------- Load Checkpoint ----------#
    #load_discrete_backbone(model, discrete_resume_path)
    resume_path = None  # Set to your resume path if needed
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
    
    # ---------- Logging ----------#
    if rank == 0:
        wandb.login(key="3761ca3994faa9fea7e291585ce72a0ed49562a0")
        logger = wandb.init(project="ContinousPrior")
    
    # Training loop
    model.train()
    for epoch in range(start_epoch, epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for batch in pbar:
            batch = batch[0].to(device)  # Move batch to the current device
            B, C, H, W = batch.shape
            batch = batch.permute(0, 2, 3, 1).reshape(B, H*W, C)
            logits, mse_loss, ploss, gt_dec, pred_dec = model(batch)
            loss = mse_loss + ploss  # Combine losses
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += mse_loss.item() * batch.size(0)
            pbar.set_postfix(loss=loss.item())
            global_step += 1
            if rank == 0:
                logger.log({"batch_total_loss": loss.item()}, step=global_step)
                logger.log({"batch_mse_loss": mse_loss.item()}, step=global_step)
                logger.log({"batch_ploss": ploss.item()}, step=global_step)                
        
        # Visualize
        if rank == 0 and (epoch+1) % vis_every == 0:
            gen_grid = make_grid(pred_dec[:vis_num], nrow=vis_num, normalize=True, value_range=(-1, 1))
            gt_grid = make_grid(gt_dec[:vis_num], nrow=vis_num, normalize=True, value_range=(-1, 1))
            combined_grid = torch.cat((gt_grid, gen_grid), dim=1)
            # Convert to NumPy format
            combined_image = combined_grid.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
            combined_image = (combined_image * 255).astype(np.uint8)  # Convert
            logger.log({"Fitting Results": wandb.Image(combined_image)}, step=global_step)
            # Save the combined image
            save_path = f"factorized_VAE/ContinousPrior/fitting_res.png"
            save_image(combined_grid, save_path, normalize=False)
            
        # Save checkpoint
        if rank == 0 and ((epoch+1) % save_every == 0 or (epoch+1) == epochs):
            ckpt_state = {
                "epoch": epoch+1,
                "global_step": global_step+1,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            ckpt_path = f"factorized_VAE/ContinousPrior/ContinousPrior_disweight_ploss_epoch{epoch+1}.pth"
            torch.save(ckpt_state, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
    print("Training complete.")
    
def run_ddp():
    world_size = torch.cuda.device_count()
    print(f"Running DDP with {world_size} GPUs")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    
if __name__ == "__main__":
    run_ddp()









