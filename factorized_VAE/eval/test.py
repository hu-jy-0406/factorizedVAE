from locale import normalize
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from factorized_VAE.my_models.fvae import FVAE
import numpy as np

model = FVAE()
ckpt_path = "/home/renderex/causal_groups/jinyuan.hu/ckpts/fvae/full/fvae_full_3loss+epoch10.pth"
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt["model"])

# transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
# dataset = ImageFolder("/home/renderex/causal_groups/jinyuan.hu/CIFAR10", transform=transform)

# test_img0, _ = dataset[0]
# test_img0 = test_img0.unsqueeze(0)
# test_img1, _ = dataset[1]
# test_img1 = test_img1.unsqueeze(0)
# test_img = torch.cat([test_img0, test_img1], dim=0)
# print(f"Test image shape: {test_img.shape}")  # (B, 3, 256, 256)
# test_img = test_img.cuda()
# test_indices = model.get_indices(test_img) # (B*256)
# print(f"Test indices shape: {test_indices.shape}")

indices_path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-code/fvae/cifar10256_codes/1.npy"
indices = np.load(indices_path)
indices = torch.tensor(indices).cuda()
print("indices.shape:", indices.shape)
test_indices = torch.flatten(indices)

test_zq = model.vq.quantize.get_codebook_entry(test_indices, (1, 16, 16, 8)) # shape specifying (batch, height, width, channel)
# (B, 8, 16, 16)
print(f"Test zq shape: {test_zq.shape}")
test_recon = model.vq.decode(test_zq)

# Save the reconstructed image
save_image(test_recon, "/home/renderex/causal_groups/jinyuan.hu/factorizedVAE/factorized_VAE/images/fvae_recon.png", normalize=True, value_range=(-1, 1))
