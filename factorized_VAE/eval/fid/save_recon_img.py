import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
#import tensorflow.compat.v1 as tf
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from factorized_VAE.my_models.lfq import LFQ_model
from factorized_VAE.my_models.autoencoder import VQModelInterface
import factorized_VAE.my_models.autoencoder as autoencoder
from evaluator import Evaluator
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
import yaml
import random
from datetime import timedelta
from imagefolder_models.vae import AutoencoderKL

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '13446'

def main(rank, world_size):
    
    model_type = "kl-8"
    
    num_samples = 60000
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #---------- DDP setup ----------#
    dist.init_process_group(backend='nccl', init_method='env://', 
                            world_size=world_size, rank=rank,
                            timeout=timedelta(seconds=10))
    torch.cuda.set_device(rank)
    print(f"Rank: {rank}, Device: {device}")
    
    if model_type == "lfq_vae":
        model = LFQ_model()
        ckpt_path = "factorized_VAE/ckpts/lfq_vae/lfq_vae_epoch10.pth"
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        recon_save_path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-recon-lfq-vae"
    elif model_type == "kl-16":
        model = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path="/home/renderex/causal_groups/jinyuan.hu/mar/pretrained_models/vae/kl16.ckpt").cuda().eval()
        recon_save_path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-recon-kl-16"
    #TODO#
    elif model_type == "kl-8":
        config_path = "factorized_VAE/ckpts/first_stage_models/kl-f8/config.yaml"
        ckpt_path = "factorized_VAE/ckpts/first_stage_models/kl-f8/model.ckpt"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        ddconfig = config["model"]["params"]["ddconfig"]
        lossconfig = config["model"]["params"]["lossconfig"]
        embed_dim = config["model"]["params"]["embed_dim"]
        model = autoencoder.AutoencoderKL(
            embed_dim=embed_dim,
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            ckpt_path=ckpt_path,
            ignore_keys=[],
        )
        recon_save_path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-recon-kl-f8"
    elif model_type == "vq-f8":
        # 加载配置文件
        config_path = "factorized_VAE/ckpts/first_stage_models/vq-f8/config.yaml"
        ckpt_path = "factorized_VAE/ckpts/first_stage_models/vq-f8/model.ckpt"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # 提取配置
        ddconfig = config["model"]["params"]["ddconfig"]
        lossconfig = config["model"]["params"]["lossconfig"]
        embed_dim = config["model"]["params"]["embed_dim"]
        n_embed = config["model"]["params"]["n_embed"]  # 提取 n_embed 参数
        # 初始化 VQModelInterface
        model = VQModelInterface(
            embed_dim=embed_dim,
            n_embed=n_embed,  # 添加 n_embed 参数
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            ckpt_path=ckpt_path,
            ignore_keys=[],
        )
        #重建图像存储路径
        recon_save_path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-recon-vq-f8"
    
    upsampled_save_path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-256-full"
    os.makedirs(upsampled_save_path, exist_ok=True)
    os.makedirs(recon_save_path, exist_ok=True)
    print(f"Saving upsampled images to {upsampled_save_path}")
    print(f"Saving recon images to {recon_save_path}")
    
    
    model = model.to(device).eval()
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    #---------- Load Dataset ----------#
    batch_size = 128
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder("/home/renderex/causal_groups/jinyuan.hu/CIFAR10", transform=transform)
    
    
    # -------- Random choose a subset -------- #
    if num_samples < len(dataset):
        print(f"Randomly selecting {num_samples} samples from the dataset")
        random.seed(42)  # For reproducibility
        total_indices = list(range(len(dataset)))
        selected_indices = random.sample(total_indices, num_samples)
        dataset = torch.utils.data.Subset(dataset, selected_indices)
        
    print(f"Dataset size: {len(dataset)}")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    
    samples = []
    
    for x, _ in tqdm(loader, disable=not rank == 0):
        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            sample = model.module.reconstruct(x)
            
            # # resize to 32x32
            # resize_to_32 = transforms.Resize((32, 32))
            # sample = resize_to_32(sample)
            
            samples.append(sample.cpu())
            
            # 修改图像存储路径
            # for i in range(sample.shape[0]):
            #     #save_image(x[i:i+1], os.path.join(upsampled_save_path, f"gt_rank{rank}_{total + i}.png"), normalize=True, value_range=(-1, 1))
            #     save_image(sample[i:i+1], os.path.join(recon_save_path, f"recon_rank{rank}_{total + i}.png"), normalize=True, value_range=(-1, 1))
            
            #total += sample.shape[0]
    
    samples = torch.cat(samples, dim=0)
    total = 0
    for i in tqdm(range(samples.shape[0]), desc="Saving images"):
        # gt_path = os.path.join(upsampled_save_path, f"gt_rank{rank}_{total + i}.png")
        recon_path = os.path.join(recon_save_path, f"recon_{total + i}.png")
        
        # 保存 Ground Truth 图像
        # save_image(x[i:i+1], gt_path, normalize=True, value_range=(-1, 1))
        
        # 保存重建图像
        save_image(samples[i:i+1], recon_path, normalize=True, value_range=(-1, 1))
        
    dist.barrier()
    dist.destroy_process_group()
    
def run_ddp():
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    
if __name__ == "__main__":
    run_ddp()
    
#把文件删了，用cuda0从头开始跑
