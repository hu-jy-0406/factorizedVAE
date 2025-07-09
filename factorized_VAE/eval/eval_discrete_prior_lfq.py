import torch
from factorized_VAE.my_models.lfq import LFQ_model
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from factorized_VAE.eval.evaluation import calculate_fid
import numpy as np
from tqdm import tqdm
import random
import os



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ------- Load LFQ-VAE Model ------- #
    #lfq_model = LFQ_model("factorized_VAE/ckpts/vae/kl16.ckpt").to(device).eval()
    lfq_model = LFQ_model().to(device).eval()
    resume_path = "factorized_VAE/ckpts/lfq_vae/lfq_vae_epoch10.pth"
    ckpt = torch.load(resume_path, map_location=device)
    lfq_model.load_state_dict(ckpt["model"])
    

    # ------- Load Generated Tokens ------- #
    gen_indices_path = "factorized_VAE/generated_tokens/all_generated_tokens.npy"
    gen_indices = np.load(gen_indices_path)
    num_samples = 60000
    gen_indices = gen_indices[:num_samples]  # shape (N, 64)
    gen_indices_tensor = torch.tensor(gen_indices, dtype=torch.long).to(device)  # shape (N, 64)
    
    
    # ------- Decode Generated Tokens ------- #
    batch_size = 32  # 根据显存大小调整批量大小
    samples = []
    for i in tqdm(range(0, num_samples, batch_size), desc="Decoding batches"):
        batch_indices = gen_indices_tensor[i:i + batch_size]  # 获取当前批次
        gen_latents = lfq_model.quantizer.indices_to_latents(batch_indices)  # shape (B, 256, 16)
        gen_latents = gen_latents.reshape(gen_latents.shape[0], 16, 16, 16)
        gen_imgs = lfq_model.decode_quantized_latent(gen_latents)  # shape should be (B, 3, 256, 256)
        
        save_image(gen_imgs, "factorized_VAE/images/discrete_prior_gen_imgs.png", nrow=8, normalize=True, value_range=(-1, 1))
        
        gen_imgs = torch.clamp(127.5 * gen_imgs + 128.0, 0, 255).to(torch.uint8).contiguous()#(B, 3, 256, 256)
        samples.append(gen_imgs.cpu().numpy())  # 将结果移到 CPU 并保存

    # 将所有批次的结果拼接成一个数组
    samples = np.concatenate(samples, axis=0)
    samples = samples.transpose(0, 2, 3, 1)


    # ------- Load Ground Truth Images ------- #
    transform = transforms.Compose([
        #upsample the images from 32*32 to 256*256
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
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
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    gt = []
    for x, _ in tqdm(loader, desc="Loading validation data"):
        # Process images for FID calculation
        # Upsample the images from 32*32 to 256*256 using transforms.Resize
        x = torch.clamp(127.5 * x + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
        gt.append(x.numpy())
    gt = np.concatenate(gt, axis=0)
    
    print(f"Sampled ground truth images shape: {samples.shape}")  # (num_samples, 256, 256, 3)
    print(f"Ground truth images shape: {gt.shape}")  # (num_samples, 256, 256, 3)

    fid, is_score = calculate_fid(samples, gt, batch_size=batch_size)
    print(f"Generation FID: {fid:.4f}")
    print(f"Inception Score: {is_score:.4f}")
    
if __name__ == "__main__":
    main()