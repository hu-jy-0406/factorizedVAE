import yaml
import torch
from factorized_VAE.my_models.autoencoder import VQModelInterface

# 加载配置文件
config_path = "factorized_VAE/ckpts/vqvae/config.yaml"
ckpt_path = "factorized_VAE/ckpts/vqvae/vq-f16.ckpt"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 提取配置
ddconfig = config["model"]["params"]["ddconfig"]
lossconfig = config["model"]["params"]["lossconfig"]
embed_dim = config["model"]["params"]["embed_dim"]
n_embed = config["model"]["params"]["n_embed"]  # 提取 n_embed 参数

# 初始化 VQModelInterface
vq_model = VQModelInterface(
    embed_dim=embed_dim,
    n_embed=n_embed,  # 添加 n_embed 参数
    ddconfig=ddconfig,
    lossconfig=lossconfig,
    ckpt_path=ckpt_path,
    ignore_keys=[],
)

# 打印模型结构以验证
print(vq_model)