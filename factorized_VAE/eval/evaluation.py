import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from evaluator import Evaluator
import tensorflow.compat.v1 as tf


def evaluation_FID(model, vq_model, val_data_path):
    #prepare dataset
    transform = transforms.Compose([
        #upsample the images from 32*32 to 256*256
        # transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(val_data_path, transform=transform)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    num_samples = len(dataset)
    print(f"Number of samples in validation dataset: {num_samples}")

    batch_size = 128  # Adjust batch size as needed

    #sample images from the model
    model.eval()
    generated_samples = []
    total = 0
    with torch.no_grad():
        for _ in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
            current_batch_size = min(batch_size, num_samples - total)
            
            # Generate tokens
            tokens = model.generate(num_samples=current_batch_size)
            
            # Process tokens to images in batch
            batch_tokens = []
            for i in range(tokens.shape[0]):
                tokens_i = tokens[i].reshape(8, 8)
                batch_tokens.append(tokens_i)
                
            # Convert to tensor and process in batch
            batch_tokens = torch.stack(batch_tokens).to("cuda")  # shape (batch_size, 8, 8)
            latents = vq_model.quantize.codebook_lookup(batch_tokens)  # shape (batch_size, 8, 8, codebook_embed_dim)
            images = vq_model.decode(latents)  # shape (batch_size, 3, 32, 32)
            
            # Process for FID calculation (following xqgan_train.py format)
            images = torch.clamp(127.5 * images + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
            generated_samples.append(images.cpu().numpy())
            
            total += current_batch_size
            if total >= num_samples:
                break
    
    # Concatenate all generated samples
    generated_samples = np.concatenate(generated_samples, axis=0)
    print(f"Generated {generated_samples.shape[0]} samples for FID calculation")   

    

    gt = []
    for x, _ in tqdm(val_loader, desc="Loading validation data"):
        # Process images for FID calculation
        x = torch.clamp(255 * x, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
        gt.append(x.numpy())
    gt = np.concatenate(gt, axis=0)

    fid, is_score = calculate_fid(generated_samples, gt)
    print(f"Generation FID: {fid:.4f}")
    print(f"Inception Score: {is_score:.4f}")


def load_validation_data(args):
    """Load CIFAR-10 validation data for reference FID calculation"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    val_dataset = datasets.ImageFolder(args.val_data_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    gt = []
    for x, _ in tqdm(val_loader, desc="Loading validation data"):
        # Process images for FID calculation
        x = torch.clamp(255 * x, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
        gt.append(x.numpy())
    
    gt = np.concatenate(gt, axis=0)
    print(f"Loaded {gt.shape[0]} validation images")
    return gt

def calculate_fid(generated_samples, real_samples, batch_size):
    """Calculate FID between generated and real samples"""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    
    evaluator = Evaluator(tf.Session(config=config), batch_size=batch_size)
    evaluator.warmup()
    
    print("Computing reference batch activations...")
    ref_acts = evaluator.read_activations(real_samples)
    print("Computing reference batch statistics...")
    ref_stats, _ = evaluator.read_statistics(real_samples, ref_acts)
    
    print("Computing generated sample activations...")
    sample_acts = evaluator.read_activations(generated_samples)
    print("Computing generated sample statistics...")
    sample_stats, _ = evaluator.read_statistics(generated_samples, sample_acts)
    
    fid = sample_stats.frechet_distance(ref_stats)
    # print(f"Generation FID: {fid:.4f}")
    
    # Calculate Inception Score
    is_score = evaluator.compute_inception_score(sample_acts[0])
    # print(f"Inception Score: {is_score:.4f}")
    
    return fid, is_score



