import torch
import torch.nn as nn
import yaml
import factorized_VAE.my_models.autoencoder as autoencoder
from factorized_VAE.my_models.adablock import AdaBlock


class FVAE(nn.Module):
    '''
    vae encoder: 
        kl-f16 (from stable-diffusion)
        input: x (B, 3, 256, 256)
        output: z (B, 16, 16, 16)
    AdaBlock (get z' from z)
        input: z (B, 16, 16, 16)
        output: z' (B, 8, 16, 16)
    vqvae quant conv:
        vq-f16 (from stable-diffusion)
        input: z" (B, 8, 16, 16)
        output: z" (B, 8, 16, 16)
    vqvae quantizer:
        vq-f16 (from stable-diffusion)
        input: z" (B, 8, 16, 16)
        output: z_q (B, 8, 16, 16), indices (B, 16, 16)
    vqvae decoder:
        vq-f16 (from stable-diffusion)
        input: z_q (B, 8, 16, 16)
        output: x_recon (B, 3, 256, 256)
    vae decoder:
        kl-f16 (from stable-diffusion)
        input: z (B, 16, 16, 16)
        output: x_recon (B, 3, 256, 256)
    '''
    def __init__(self):
        super().__init__()
        self.kl = self.load_autoencoder(
            config_path="/home/guangyi.chen/causal_group/jinyuan.hu/ckpts/first_stage_models/kl-f16/config.yaml",
            ckpt_path="/home/guangyi.chen/causal_group/jinyuan.hu/ckpts/first_stage_models/kl-f16/model.ckpt",
            type="kl"
        ).cuda().eval()
        self.vq = self.load_autoencoder(
            config_path="/home/guangyi.chen/causal_group/jinyuan.hu/ckpts/first_stage_models/vq-f16/config.yaml",
            ckpt_path="/home/guangyi.chen/causal_group/jinyuan.hu/ckpts/first_stage_models/vq-f16/vq-f16.ckpt",
            type="vq"
        ).cuda().eval()
        self.ada_block = AdaBlock(in_channels=16, out_channels=8).cuda()
        for param in self.kl.parameters():
            param.requires_grad = False
        for param in self.vq.parameters():
            param.requires_grad = False
    
    def load_autoencoder(self, config_path, ckpt_path, type):
        '''
        Load autoencoder from config and ckpt path
        type: "kl" or "vq"
        '''
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        ddconfig = config["model"]["params"]["ddconfig"]
        lossconfig = config["model"]["params"]["lossconfig"]
        embed_dim = config["model"]["params"]["embed_dim"]
        if type == "kl":
            model = autoencoder.AutoencoderKL(
                embed_dim=embed_dim, ddconfig=ddconfig, lossconfig=lossconfig,
                ckpt_path=ckpt_path, ignore_keys=[],
            )
        elif type == "vq":
            n_embed = config["model"]["params"]["n_embed"]
            model = autoencoder.VQModel(
                embed_dim=embed_dim, ddconfig=ddconfig, lossconfig=lossconfig,
                ckpt_path=ckpt_path, ignore_keys=[], n_embed=n_embed
            )
        else:
            raise ValueError(f"Unknown autoencoder type: {type}")
        return model
    
    def forward(self, x):
        '''
        for training adaptive block
        encode with kl and ada_block
        quantize with vq
        decode with vq decoder
        '''
        posterior = self.kl.encode(x)
        z = posterior.mean
        z = self.ada_block(z)
        
        z_vq = self.vq.encoder(x)
        latent_loss = torch.nn.MSELoss()(z, z_vq)
        
        h = self.vq.quant_conv(z)
        quant, _, _ = self.vq.quantize(h)
        
        dec = self.vq.decode(quant)
        
        return dec, latent_loss
    
    def reconstruct(self, x):
        '''
        for inference
        encode with kl and ada_block
        quantize with vq
        decode with vq decoder
        '''
        posterior = self.kl.encode(x)
        z = posterior.mean
        z = self.ada_block(z)
        
        h = self.vq.quant_conv(z)
        quant, _, _ = self.vq.quantize(h)
        quant = self.vq.post_quant_conv(quant)
        dec = self.vq.decoder(quant)
        
        return dec
    
    def get_indices_from_images(self, x):
        '''
        get quantized indices from input image x
        x.shape = (batch_size, 3, 256, 256)
        indices.shape = (batch_size*16*16)
        '''
        z = self.get_vae_latent_from_images(x)
        z = self.ada_block(z)
        
        h = self.vq.quant_conv(z)
        _, _, (_, _, indices) = self.vq.quantize(h)
        return indices
    
    def get_vae_latent_from_images(self, x):
        '''
        get vae latent z from input image x
        x.shape = (batch_size, 3, 256, 256)
        z.shape = (batch_size, 16, 16, 16)
        '''
        posterior = self.kl.encode(x)
        z = posterior.mean
        return z
    
    def get_quant_from_vae_latent(self, z):
        '''
        get quantized latent from vae latent z
        z.shape = (batch_size, 16, 16, 16)
        quant.shape = (batch_size, 8, 16, 16)
        '''
        #print("z.shape:", z.shape)
        z = self.ada_block(z)
        #print("z.shape:", z.shape)
        h = self.vq.quant_conv(z)
        #print("h shape:", h.shape)
        quant, _, _ = self.vq.quantize(h)
        #print("quant shape:", quant.shape)
        return quant
    
    def recon_from_indices(self, indices):
        '''
        reconstruct from quantized indices
        indices.shape = (batch_size, 16, 16)
        output shape = (batch_size, 3, 256, 256)
        '''
        B, H, W = indices.shape
        indices = torch.flatten(indices)
        z_q = self.vq.quantize.get_codebook_entry(indices, shape=(B, H, W, 8))
        dec = self.vq.decode(z_q)
        return dec