import torch
from torchvision.utils import save_image
from factorized_VAE.my_models.lfq import LFQ_model
from factorized_VAE.my_models.discrete_prior import DiscretePrior
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "6"  # Set visible GPUs for DDP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------- Load LFQ-VAE Model ------- #
#lfq_model = LFQ_model("factorized_VAE/ckpts/vae/kl16.ckpt").to(device).eval()
lfq_model = LFQ_model().to(device).eval()
resume_path = "factorized_VAE/ckpts/lfq_vae/lfq_vae_epoch1.pth"
ckpt = torch.load(resume_path, map_location=device)
lfq_model.load_state_dict(ckpt["model"])

# # ------- Load Generated Tokens ------- #
# gen_indices_path = "factorized_VAE/generated_tokens/all_generated_tokens.npy"
# gen_indices = np.load(gen_indices_path)
# print("gen_indices_shape:", gen_indices.shape)
# num_samples = gen_indices.shape[0]
# gen_indices_tensor = torch.tensor(gen_indices, dtype=torch.long).to(device)  # shape (N, 64)

# # ------- Decode Generated Tokens ------- #
# batch_size = 20
# batch_indices = gen_indices_tensor[:batch_size]  # 获取前一个批次
# gen_latents = lfq_model.quantizer.indices_to_latents(batch_indices)  # shape (B, 64, 16)
# gen_latents = gen_latents.reshape(gen_latents.shape[0], 16, 16, 16).permute(0, 3, 1, 2)  # reshape to (B, 16, 16, 16)
# gen_imgs = lfq_model.decode_quantized_latent(gen_latents)  #
# save_image(gen_imgs, "factorized_VAE/images/discrete_prior_samples.png", nrow=5, normalize=True, value_range=(-1, 1))

# --------- Load Discrete Prior --------- #
vocab_size = 65536  # set this to match your codebook size
seq_len = 256
model = DiscretePrior(vocab_size, seq_len=seq_len, d_model=512, nhead=8, num_layers=8).to(device)
resume_path = "factorized_VAE/DiscretePrior_CIFAR_epoch2000.pth"
ckpt = torch.load(resume_path, map_location=device)
model.load_state_dict(ckpt["model"])
tokens = model.generate(num_samples=20)
print("Generated tokens shape:", tokens.shape)  # 应该是 (20, 256)
latens = lfq_model.quantizer.indices_to_latents(tokens)  # shape (20, 256, 16)
print("Latents shape:", latens.shape)  # 应该是 (20, 256, 16)
latens = latens.reshape(latens.shape[0], 16, 16, 16)  # reshape to (20, 16, 16, 16)
print("Reshaped latents shape:", latens.shape)  # 应该是 (20, 16, 16, 16)
gen_imgs = lfq_model.decode_quantized_latent(latens)  # shape should be (20, 3, 32, 32)
print("Generated images shape:", gen_imgs.shape)  # 应该是 (20, 3, 32, 32)
save_image(gen_imgs, "factorized_VAE/images/discrete_prior_samples_2000.png", nrow=5, normalize=True, value_range=(-1, 1))