import torch
import torch.nn as nn
import yaml
from factorized_VAE.my_models.autoencoder import VQModel
from imagefolder_models.vae import AutoencoderKL


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
model = VQModel(
    embed_dim=embed_dim,
    n_embed=n_embed,  # 添加 n_embed 参数
    ddconfig=ddconfig,
    lossconfig=lossconfig,
    ckpt_path=ckpt_path,
    ignore_keys=[],
)

class FVAE(nn.Module):
    '''
    pretrained vae encoder (get continous latent z from img)
    AdaBlock (get z' from z)
    pretrained vector quantizer (get z_q from z')
    pretrained vector quantizer decoder (decode z_q to x_recon)
    '''
    def __init__(self, vae_ckpt_path=None, vqvae_config=None, vqvae_ckpt_path=None):
        super().__init__()
        self.embed_dim = 16
        self.vae = AutoencoderKL(embed_dim=self.embed_dim, ch_mult=(1, 1, 2, 2, 4), 
                                 ckpt_path=vae_ckpt_path).cuda().eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae = self.load_vqvae(vqvae_config, vqvae_ckpt_path)
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def load_vqvae(self, config_path, ckpt_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # 提取配置
        ddconfig = config["model"]["params"]["ddconfig"]
        lossconfig = config["model"]["params"]["lossconfig"]
        embed_dim = config["model"]["params"]["embed_dim"]
        n_embed = config["model"]["params"]["n_embed"]  # 提取 n_embed 参数
        # 初始化 VQModelInterface
        model = VQModel(
            embed_dim=embed_dim,
            n_embed=n_embed,  # 添加 n_embed 参数
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            ckpt_path=ckpt_path,
            ignore_keys=[],
        )
        return model