import torch
import torch.nn as nn
import numpy as np
from factorized_VAE.my_models.pos_emb import SinusoidalPositionalEmbedding
from factorized_VAE.utils import process_image
from factorized_VAE.my_models.discrete_prior import DiscretePrior
from factorized_VAE.my_models.lfq import LFQ_quantizer
from tokenizer.tokenizer_image.lpips import LPIPS
from PIL import Image


class ContinousPrior(nn.Module):
    """
    Autoregressive Transformer using nn.TransformerDecoder.
    Predict token i given [CLS] and tokens < i.
    """
    def __init__(self, quantizer, vae, z_dim=32, seq_len=64, d_model=512, nhead=8, num_layers=8, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.z_dim = z_dim
        for param in quantizer.parameters():
            param.requires_grad = False
        for param in vae.parameters():
            param.requires_grad = False
        self.quantizer = quantizer
        self.vae = vae
        self.use_perceptual_loss = True  # Use perceptual loss by default
        if self.use_perceptual_loss:
            self.perceptual_loss = LPIPS().eval()
            self.perceptual_weight = 1
        else:
            self.perceptual_loss = None
            self.perceptual_weight = 0.0
        
        # Embeddings + sinusoidal positional embeddings
        self.input_proj = nn.Linear(z_dim, d_model)
        self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_len=seq_len)

        # TransformerDecoder
        backbone_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True
        )
        self.backbone = nn.TransformerDecoder(backbone_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, z_dim)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        
        B, L, C = x.shape
        
        # Validate dimensions
        if L != self.seq_len:
            raise ValueError(f"Input sequence length {L} does not match model sequence length {self.seq_len}.")
        if C != self.z_dim:
            raise ValueError(f"Input feature dimension {C} does not match model z_dim {self.z_dim}.")
        
        start_token = torch.zeros((B, 1, C), device=x.device)  # Start token for context
        
        # Initialize predictions container
        #predictions = torch.zeros_like(x)
        predictions = []
        x_origin = x.clone()  # Keep a copy of the original input for memory quantization
        x = torch.cat([start_token, x], dim=1)  # Add start token to the beginning of the sequence
        
        for i in range(L):
            context = x[:, :i+1, :]
            
            # Project context to model dimension
            context_proj = self.input_proj(context)  # (B, i+1, d_model)
            
            # Add positional embeddings
            context_pos = self.pos_emb(context_proj)  # (B, i+1, d_model)
            
            # Get memory for position i
            mem_i = x[:, i+1:i+2, :].unsqueeze(2).permute(0,3,1,2)  # (B, 1, z_dim)
            mem_i,_ = self.quantizer.forward(mem_i) # Quantize memory token
            mem_i = mem_i.permute(0,2,3,1)
            mem_i = mem_i.squeeze(2).detach()
            mem_i_proj = self.input_proj(mem_i)  # (B, 1, d_model)
            zero_memory = torch.zeros_like(mem_i_proj)  # shape (B, 1, d_model)
            
            # Run decoder
            # No need for causal mask since we're only providing valid context
            out = self.backbone(
                tgt=context_pos,
                memory=mem_i_proj,  # Memory for position i
                tgt_mask=None  # Context is already properly limited
            )
            
            # Get prediction for position i
            pred = self.output_proj(self.ln(out))  # (B, C)
            pred_i = pred[:, -1:, :]  # Get the last prediction (for position i)

            predictions.append(pred_i)  # Append the prediction for position i

        predictions = torch.cat(predictions, dim=1)  # shape (B, L, z_dim)
        mse_loss = nn.MSELoss()(predictions, x_origin)        

        B, L, C = predictions.shape
        W = int(np.sqrt(self.seq_len))
        predictions = predictions.reshape(B, W, W, C).permute(0, 3, 1, 2)
        pred_dec = self.vae.decode(predictions)
        x_origin = x_origin.reshape(B, W, W, C).permute(0, 3, 1, 2)
        gt_dec = self.vae.decode(x_origin)

        if self.use_perceptual_loss:
            ploss = self.perceptual_loss(gt_dec.to(self.device), pred_dec.to(self.device)).mean()  # Calculate perceptual loss
            ploss = self.perceptual_weight * ploss  # Add perceptual loss to the
        else:
            ploss = torch.tensor(0.0, device=self.device)      

        return predictions, mse_loss, ploss, gt_dec, pred_dec

    @torch.no_grad()
    def get_next_latent(self, latents, next_token):
        B, L, C = latents.shape
        # #codebook_lookup
        memory = self.quantizer.indices_to_latents(next_token)  # shape (B, 1, z_dim)
        memory_proj= self.input_proj(memory)  # Project to d_model
        # zero_memory = torch.zeros_like(memory_proj)  # shape (B, 1, d_model)  
        h = self.pos_emb((self.input_proj(latents)))  # shape (B, L, d_model)
        out = self.backbone(tgt=h, memory=memory_proj, tgt_mask=None)
        logits = self.output_proj(self.ln(out))
        last_logits = logits[:, -1, :]
        return last_logits.unsqueeze(1)

    ##TODO##
    @torch.no_grad()
    def generate(self, discrete_prior, num_samples):
        
        token_list = []
        latent_list = []

        start_token = torch.zeros((num_samples,1)).long().to(self.device)
        start_latent = torch.zeros((num_samples, 1, self.z_dim), device=self.device)  # Start with zero vector

        token_list.append(start_token)  # Append start token ID (assumed to be 0)
        latent_list.append(start_latent)  # Append start token

        for i in range(self.seq_len):
            tokens = torch.cat(token_list, dim=1).to(self.device)
            latents = torch.cat(latent_list, dim=1).to(self.device)
            #print(f"tokens shape: {tokens.shape}, latents shape: {latents.shape}")

            next_token = discrete_prior.get_next_token(tokens)
            next_latent = self.get_next_latent(latents, next_token)
            #next_latent = torch.rand_like(next_latent)  # Randomly initialize the next latent vector
            #next_latent = self.vq_model.quantize.codebook_lookup_flat(next_token)  # shape (num_samples, 1, z_dim)

            token_list.append(next_token)  # Update the next token in the sequence
            latent_list.append(next_latent)  # Update the next latent in the sequence

        tokens = torch.cat(token_list, dim=1)  # shape (num_samples, seq_len+1)
        latents = torch.cat(latent_list, dim=1)  # shape (num_samples, seq_len, z_dim)

        tokens = tokens[:, 1:]  # Remove the start token
        latents = latents[:, 1:, :]  # Remove the start token
        #latents = torch.rand_like(latents)  # Randomly initialize the latents

        return tokens, latents