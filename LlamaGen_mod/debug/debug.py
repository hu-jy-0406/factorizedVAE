import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import lpips  # pip install lpips

def load_image(path, image_size=256):
    img = Image.open(path).convert('RGB')
    img = img.resize((image_size, image_size))
    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

def compute_l2_loss(img1, img2):
    return F.mse_loss(img1, img2)

def compute_perceptual_loss(img1, img2, loss_fn):
    # lpips要求输入范围[-1,1]，shape为(N,3,H,W)
    return loss_fn(img1, img2)

if __name__ == "__main__":
    path1 = "/home/guangyi.chen/causal_group/jinyuan.hu/factorizedVAE/LlamaGen_mod/samples/GPT-Reg-B-0000100-cfg-2.0-seed-0-debug/000000_gt.png"
    #path2 = "/home/guangyi.chen/causal_group/jinyuan.hu/factorizedVAE/LlamaGen_mod/samples/GPT-Reg-B-0000100-cfg-2.0-seed-0-debug/000000.png"
    #path2 = "/home/guangyi.chen/causal_group/jinyuan.hu/factorizedVAE/LlamaGen_mod/samples/GPT-Reg-B-0000500-cfg-2.0-seed-0-debug/000000.png"
    path2 = "/home/guangyi.chen/causal_group/jinyuan.hu/factorizedVAE/LlamaGen_mod/samples/GPT-Reg-B-0026000-cfg-2.0-seed-0-debug/000000.png"
    image_size = 256

    img1 = load_image(path1, image_size).cuda()
    img2 = load_image(path2, image_size).cuda()

    l2_loss = compute_l2_loss(img1, img2)

    loss_fn = lpips.LPIPS(net='vgg').cuda()
    perceptual_loss = compute_perceptual_loss(img1, img2, loss_fn)

    print(f"L2 Loss: {l2_loss.item():.6f}")
    print(f"Perceptual Loss: {perceptual_loss.item():.6f}")

# 100 steps
# L2 Loss: 0.007022
# Perceptual Loss: 0.195296

# 500 steps
# L2 Loss: 0.001952
# Perceptual Loss: 0.100539

# 26000 steps
# L2 Loss: 0.000845
# Perceptual Loss: 0.048377