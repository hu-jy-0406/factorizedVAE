import torch

# 加载原始 ckpt 文件
original_ckpt_path = "factorized_VAE/ckpts/vae/kl16.ckpt"
new_ckpt_path = "factorized_VAE/ckpts/vae/kl16_encoder_quant_conv.ckpt"

# 加载原始 ckpt
ckpt = torch.load(original_ckpt_path, map_location="cpu")

# 提取 encoder 和 quant_conv 的权重，并去掉前缀
new_ckpt = {"encoder": {}, "quant_conv": {}}
for key, value in ckpt["model"].items():
    if key.startswith("encoder."):
        new_key = key[len("encoder."):]  # 去掉 "encoder." 前缀
        new_ckpt["encoder"][new_key] = value
    elif key.startswith("quant_conv."):
        new_key = key[len("quant_conv."):]  # 去掉 "quant_conv." 前缀
        new_ckpt["quant_conv"][new_key] = value

# 保存新的 ckpt 文件
torch.save(new_ckpt, new_ckpt_path)
print(f"New checkpoint saved to {new_ckpt_path}")