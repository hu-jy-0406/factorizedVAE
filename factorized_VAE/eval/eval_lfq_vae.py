import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import tensorflow.compat.v1 as tf
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from factorized_VAE.my_models.lfq import LFQ_model
from evaluator import Evaluator
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
import random
from datetime import timedelta
from imagefolder_models.vae import AutoencoderKL

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12346'


def main(rank, world_size):
    
    num_samples = 60000
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #---------- DDP setup ----------#
    dist.init_process_group(backend='nccl', init_method='env://', 
                            world_size=world_size, rank=rank,
                            timeout=timedelta(seconds=10))
    torch.cuda.set_device(rank)
    print(f"Rank: {rank}, Device: {device}")
    
    lfq_model = LFQ_model()
    ckpt_path = "factorized_VAE/ckpts/lfq_vae/lfq_vae_epoch10.pth"
    ckpt = torch.load(ckpt_path)
    lfq_model.load_state_dict(ckpt['model'])
    lfq_model.eval()
    
    #vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path="/home/renderex/causal_groups/jinyuan.hu/mar/pretrained_models/vae/kl16.ckpt").cuda().eval()

    
    model = DDP(lfq_model, device_ids=[rank], output_device=rank)
    
    #---------- Load Dataset ----------#
    batch_size = 64
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
    
    total = 0
    samples = []
    gt = []
    vis_num = 4  # Number of images to visualize
    
    for x, _ in tqdm(loader, disable=not rank == 0):
        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            sample = model.module.reconstruct(x)
            
            gen_grid = make_grid(sample[:vis_num], nrow=vis_num, normalize=True, value_range=(-1, 1))
            gt_grid = make_grid(x[:vis_num], nrow=vis_num, normalize=True, value_range=(-1, 1))
            combined_grid = torch.cat((gt_grid, gen_grid), dim=1)
            save_path = f"factorized_VAE/images/lfq_decoder_recon.png"
            save_image(combined_grid, save_path, normalize=False)
            
            sample = torch.clamp(127.5 * sample + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
            x = torch.clamp(127.5 * x + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()

        sample = torch.cat(dist.nn.all_gather(sample), dim=0)
        x = torch.cat(dist.nn.all_gather(x), dim=0)
        samples.append(sample.to("cpu", dtype=torch.uint8).numpy())
        gt.append(x.to("cpu", dtype=torch.uint8).numpy())
        
        total += sample.shape[0]
    #model.train()
    print(f"Ealuate total {total} files.")
    dist.barrier()
    if rank == 0:
        samples = np.concatenate(samples, axis=0)
        gt = np.concatenate(gt, axis=0)
        config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True
        evaluator = Evaluator(tf.Session(config=config), batch_size=32)
        evaluator.warmup()
        print("computing reference batch activations...")
        ref_acts = evaluator.read_activations(gt)
        print("computing/reading reference batch statistics...")
        ref_stats, _ = evaluator.read_statistics(gt, ref_acts)
        print("computing sample batch activations...")
        sample_acts = evaluator.read_activations(samples)
        print("computing/reading sample batch statistics...")
        sample_stats, _ = evaluator.read_statistics(samples, sample_acts)
        FID = sample_stats.frechet_distance(ref_stats)
        print(f"FID {FID:07f}")
        # # eval code, delete prev if not the best
        # if curr_fid == None:
        #     curr_fid = [FID, train_steps]
        # elif FID <= curr_fid[0]:
        #     # os.remove(f"{cloud_checkpoint_dir}/{curr_fid[1]:07d}.pt")
        #     curr_fid = [FID, train_steps]
    dist.barrier()
    dist.destroy_process_group()
    
def run_ddp():
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    
if __name__ == "__main__":
    run_ddp()
