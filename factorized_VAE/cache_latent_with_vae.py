import torch
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from imagefolder_models.vae import AutoencoderKL
from factorized_VAE.utils import process_image, recon_img_with_vae
from factorized_VAE.my_models.lfq import LFQ_quantizer

#export PYTHONPATH=/home/renderex/causal_groups/jinyuan.hu/factorizedVAE:$PYTHONPATH

vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path="/home/renderex/causal_groups/jinyuan.hu/mar/pretrained_models/vae/kl16.ckpt").cuda().eval()

#image_path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10/train/airplane/10009.png"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# -------------------cache latent-------------------#
# # 使用 ImageFolder 加载 CIFAR10 所有图片（训练集和验证集需放在同一目录下不同子文件夹中）
# dataset = ImageFolder("/home/renderex/causal_groups/jinyuan.hu/CIFAR10", transform=transform)
# print(f"Dataset size: {len(dataset)}")
# dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

# latents_list = []
# with torch.no_grad():
#     for images, _ in tqdm(dataloader):
#         images = images.cuda()
#         posterior = vae.encode(images)
#         z = posterior.mean  # 提取隐变量均值
#         latents_list.append(z.cpu())

# # 将所有隐变量按顺序串联，shape = [N, latent_dim]
# latents = torch.cat(latents_list, dim=0)
# print(f"Latents shape: {latents.shape}")

# # 保存到文件，后续可直接加载构建数据集和 DataLoader
# latent_save_path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-VAE-latent/CIFAR10-VAE-latents.pt"
# torch.save(latents, latent_save_path)
# print(f"Saved latents shape: {latents.shape} to {latent_save_path}")

# -------------------load latent-------------------#
latent_save_path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-VAE-latent/CIFAR10-VAE-latents.pt"
latents = torch.load(latent_save_path)
print(f"Loaded latents shape: {latents.shape}")
dataset = TensorDataset(latents)
# 构建 DataLoader，可用于后续生成模型的训练
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
print(f"Loaded latents dataset shape: {latents.shape}")
# z = latents[1].unsqueeze(0).cuda()  # 取第一个 latent
# print(f"Latent z shape: {z.shape}")
# recon_img_from_vae_latent(vae, z, save_path="/home/renderex/causal_groups/jinyuan.hu/factorizedVAE/factorized_VAE/images/recon.png")

#-------------------quantize latent-------------------#

# 对所有 latent 做量化
all_indices = []  # 用于存储所有样本的索引
dataloader = DataLoader(latents, batch_size=1, shuffle=False)
quantizer = LFQ_quantizer(embedding_dim=16).cuda()  # 假设 embedding_dim 为 16
quantizer.eval()
with torch.no_grad():
    for batch in tqdm(dataloader):
        batch = batch.cuda()  # 假设使用 GPU
        latent, indices = quantizer(batch)
        #latents_recon = quantizer.indices_to_latents(indices)  # 将索引转换回 latent
        all_indices.append(indices.cpu())

# 将所有样本的索引拼接起来
indices_all = torch.cat(all_indices, dim=0)
print("indices_all.shape:", indices_all.shape)
save_indices_path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-VAE-latent/CIFAR10-VAE-discrete-indices.pt"
torch.save(indices_all, save_indices_path)
print(f"Saved discrete indices shape: {indices_all.shape} to {save_indices_path}")