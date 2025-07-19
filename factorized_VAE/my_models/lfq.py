import torch
import torch.nn as nn
from itertools import product
from imagefolder_models.vae import AutoencoderKL
from tokenizer.tokenizer_image.lpips import LPIPS



class LFQ_quantizer(nn.Module):
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.num_embeddings = 2 ** embedding_dim
        self.embedding_dim = embedding_dim
        weight = torch.tensor(list(product((-1.0, 1.0), repeat=embedding_dim)))
        self.register_buffer("weight", weight)
        self.register_buffer("place_values", 2 ** torch.arange(embedding_dim).flip(-1))
        
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).to(x.dtype) * 2.0 - 1.0
    
    def indices_to_latents(self, indices: torch.Tensor) -> torch.Tensor:
        return self.weight[indices]
    
    def latents_to_indices(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.place_values * (latents > 0), dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized = self.quantize(x)
        indices = self.latents_to_indices(quantized)
        return quantized, indices
    
class LFQ_model(nn.Module):
    
    '''
    Encoder from AutoencoderKL used by MAR, with parameters fixed
    Look-up-Free Quantizer from https://gist.github.com/4c10e6f2f616f67b68d046490d02696c.git
    Decoder also from AutoencoderKL, but tuned to decode quantized latents
    
    It can be build like FAVE with both vae-decoder and vq-vae-decoder,
    but to save cuda memory during training, we only keep the vq-vae-decoder.
    '''
    
    def __init__(self, vae_ckpt_path=None):
        super().__init__()
        self.embedding_dim = 16
        self.ch_mult = (1, 1, 2, 2, 4)
        self.vae = AutoencoderKL(embed_dim=self.embedding_dim, ch_mult=(1, 1, 2, 2, 4), 
                                 ckpt_path=vae_ckpt_path).cuda()
        for param in self.vae.encoder.parameters():
            param.requires_grad = False
        for param in self.vae.quant_conv.parameters():
            param.requires_grad = False
        
        self.quantizer = LFQ_quantizer(self.embedding_dim).cuda()
        
        self.ploss = LPIPS().cuda().eval()  # Perceptual loss for reconstruction quality
        
    def load_encoder_quant_conv(self):
        '''
        Load encoder and quant_conv weights from the checkpoint
        '''
        ckpt = torch.load(self.encoder_quant_conv_ckpt_path)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.quant_conv.load_state_dict(ckpt["quant_conv"])
        
    def decode_quantized_latent(self, z):
        return self.vae.decode(z)
    
    def reconstruct(self, x):
        '''
        reconstruct with quantized latent
        '''
        posterior = self.vae.encode(x) #x.shape = (batch_size, 3, 256, 256)
        z = posterior.mean
        z_q, _ = self.quantizer(z)
        x_recon = self.vae.decode(z_q)
        return x_recon
    
    def get_indices(self, x):
        '''
        get quantized indices from input image x
        x.shape = (batch_size, 3, 256, 256)
        indices.shape = (batch_size, 16, 16)
        '''
        posterior = self.vae.encode(x) #x.shape = (batch_size, 3, 256, 256)
        z = posterior.mean
        _, indices = self.quantizer(z)
        return indices        
        
    def forward(self, x):
        '''
        Encode with VAE, quantize the latent, decode the quantized latent
        '''
        posterior = self.vae.encode(x) #x.shape = (batch_size, 3, 256, 256)
        z = posterior.mean
        z_q, _ = self.quantizer(z)
        x_recon = self.vae.decode(z_q)
        recon_loss = nn.MSELoss()(x_recon, x)
        ploss = self.ploss(x_recon, x).mean()
        return recon_loss, ploss, z_q, x_recon
        

    