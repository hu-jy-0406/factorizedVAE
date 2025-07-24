# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
import argparse
import os
import yaml

from LlamaGen.utils.distributed import init_distributed_mode
from LlamaGen.dataset.augmentation import center_crop_arr
from LlamaGen.dataset.build import build_dataset
from LlamaGen.tokenizer.tokenizer_image.vq_model import VQ_models

from factorized_VAE.my_models.lfq import LFQ_model
from imagefolder_models.vae import AutoencoderKL
from factorized_VAE.my_models.fvae import FVAE
import factorized_VAE.my_models.autoencoder as autoencoder



#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Setup DDP:
    if not args.debug:
        init_distributed_mode(args)
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        device = 'cuda'
        rank = 0
    
    # Setup a feature folder:
    if args.debug or rank == 0:
        os.makedirs(args.code_path, exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_codes'), exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_labels'), exist_ok=True)

    # ----------create and load model----------#
    # vq_model = VQ_models[args.vq_model](
    #     codebook_size=args.codebook_size,
    #     codebook_embed_dim=args.codebook_embed_dim)
    # vq_model.to(device)
    # vq_model.eval()
    # checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    # vq_model.load_state_dict(checkpoint["model"])
    # del checkpoint
    model_type = args.modeltype
    if model_type == "lfq_vae":
        model = LFQ_model()
        ckpt_path = "/home/renderex/causal_groups/jinyuan.hu/ckpts/lfq_vae/lfq_vae_epoch10.pth"
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        recon_save_path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-recon-lfq-vae"
    elif model_type == "kl-16":
        model = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path="/home/renderex/causal_groups/jinyuan.hu/mar/pretrained_models/vae/kl16.ckpt").cuda().eval()
        recon_save_path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-recon-kl-16"
    elif model_type == "fvae":
        model = FVAE()
        ckpt_path = "/home/renderex/causal_groups/jinyuan.hu/ckpts/fvae/full/fvae_full_3loss+epoch10.pth"
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
    #TODO#
    # elif model_type == "kl-f8":
    #     config_path = "factorized_VAE/ckpts/first_stage_models/kl-f8/config.yaml"
    #     ckpt_path = "factorized_VAE/ckpts/first_stage_models/kl-f8/model.ckpt"
    #     with open(config_path, "r") as f:
    #         config = yaml.safe_load(f)
    #     ddconfig = config["model"]["params"]["ddconfig"]
    #     lossconfig = config["model"]["params"]["lossconfig"]
    #     embed_dim = config["model"]["params"]["embed_dim"]
    #     model = autoencoder.AutoencoderKL(
    #         embed_dim=embed_dim,
    #         ddconfig=ddconfig,
    #         lossconfig=lossconfig,
    #         ckpt_path=ckpt_path,
    #         ignore_keys=[],
    #     )
    #     recon_save_path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-recon-kl-f8"
    # elif model_type == "vq-f8":
    #     # 加载配置文件
    #     config_path = "factorized_VAE/ckpts/first_stage_models/vq-f8/config.yaml"
    #     ckpt_path = "factorized_VAE/ckpts/first_stage_models/vq-f8/model.ckpt"
    #     with open(config_path, "r") as f:
    #         config = yaml.safe_load(f)
    #     # 提取配置
    #     ddconfig = config["model"]["params"]["ddconfig"]
    #     lossconfig = config["model"]["params"]["lossconfig"]
    #     embed_dim = config["model"]["params"]["embed_dim"]
    #     n_embed = config["model"]["params"]["n_embed"]  # 提取 n_embed 参数
    #     # 初始化 VQModelInterface
    #     model = autoencoder.VQModelInterface(
    #         embed_dim=embed_dim,
    #         n_embed=n_embed,  # 添加 n_embed 参数
    #         ddconfig=ddconfig,
    #         lossconfig=lossconfig,
    #         ckpt_path=ckpt_path,
    #         ignore_keys=[],
    #     )
    #     recon_save_path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-recon-vq-f8"
    
    # Setup data:
    if args.ten_crop:
        crop_size = int(args.image_size * args.crop_range)
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.TenCrop(args.image_size), # this is a tuple of PIL Images
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    else:
        crop_size = args.image_size 
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    dataset = build_dataset(type=args.dataset_type, args=args, transform=transform)
    if not args.debug:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=args.global_seed
        )
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=1, # important!
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        #目前未使用数据增强，如有需要可以再加
        
        with torch.no_grad():
            latents = model.get_vae_latent_from_images(x) # (1, 16, 16, 16), (b, c, h, w)
            b, c, h, w = latents.shape
            codes = latents.permute(0, 2, 3, 1).reshape(b, h*w, c)  # (1, 16*16, c)       
            
        x = codes.squeeze(0).detach().cpu().numpy()    # (1, num_aug, args.image_size//16 * args.image_size//16)
        train_steps = rank + total
        np.save(f'{args.code_path}/{args.dataset}{args.image_size}_codes/{train_steps}.npy', x)

        y = y.squeeze(0).detach().cpu().numpy()    # (1,)
        np.save(f'{args.code_path}/{args.dataset}{args.image_size}_labels/{train_steps}.npy', y)
        if not args.debug:
            total += dist.get_world_size()
        else:
            total += 1
        print(total)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--modeltype", type=str, choices=["kl-f8", "vq-f8", "lfq_vae", "kl-16", "fvae"])
    parser.add_argument("--vq-ckpt", type=str, required=True, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--dataset-type", type=str, default='train')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--ten-crop", action='store_true', help="whether using random crop")
    parser.add_argument("--crop-range", type=float, default=1.1, help="expanding range of center crop")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
