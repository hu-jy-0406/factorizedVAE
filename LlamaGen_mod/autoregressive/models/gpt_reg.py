from builtins import print
from calendar import c
from doctest import OutputChecker
import torch
from torch import Tensor
from torch.nn import Dropout, Linear, MSELoss
from LlamaGen_mod.autoregressive.models.gpt import TransformerBlock, Transformer, ModelArgs, CrossAttention
from factorized_VAE.my_models.fvae import FVAE
from tokenizer.tokenizer_image.lpips import LPIPS
from typing import Optional
import numpy as np
from tqdm import tqdm

# for debug
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

                
class TransformerReg(Transformer):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        
        self.fvae = FVAE()
        ckpt_path = "/home/guangyi.chen/causal_group/jinyuan.hu/ckpts/fvae/fvae_full_3loss+epoch10.pth"
        ckpt = torch.load(ckpt_path)
        self.fvae.load_state_dict(ckpt["model"])    
        for param in self.fvae.parameters():
            param.requires_grad = False
        del ckpt
        
        del self.tok_embeddings   
        self.input_proj = Linear(self.fvae.kl.embed_dim, config.dim)
        self.input_proj2 = Linear(self.fvae.vq.embed_dim, config.dim)
        if config.no_mem:
            for param in self.input_proj2.parameters():
                param.requires_grad = False
        
        del self.output
        self.output_proj = Linear(config.dim, self.fvae.kl.embed_dim, bias=False)
        
        #self.cross_attention = CrossAttention(config)
        
        # loss
        self.mse_loss = MSELoss(reduction='mean')
        self.use_perceptual_loss = True  # Use perceptual loss by default
        if self.use_perceptual_loss:
            self.perceptual_loss = LPIPS().eval()
            for param in self.perceptual_loss.parameters():
                param.requires_grad = False
            self.perceptual_weight = 1
        else:
            self.perceptual_loss = None
            self.perceptual_weight = 0.0 
            
    def forward(
        self, 
        codes: torch.Tensor,
        cond_idx: torch.Tensor,
        mem: torch.Tensor,
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ):
        
        #mem = mem.detach()

        if codes is not None and cond_idx is not None:
            # codes.shape=(bs, 256, 16)
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num] #(bs, 1, 768)
            
            codes_embeddings = self.input_proj(codes[:, :-1, :]) # (bs, 255, 768)
            
            #print(f"codes_embeddings.shape: {codes_embeddings.shape}, cond_embeddings.shape: {cond_embeddings.shape}")
            codes_embeddings = torch.cat((cond_embeddings, codes_embeddings), dim=1) # (64, 256, 768)
            
            h = self.tok_dropout(codes_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
        else:
            if cond_idx is not None:
                #self.cls_token_num=1
                codes_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
                #codes_embeddings.shape=(64, 1, 768)
            else:
                codes_embeddings = self.input_proj(codes)
                
            bs = codes_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(codes_embeddings)  # (B, L, C)
            self.freqs_cis = self.freqs_cis        
        
        # self.freqs_cis.shape: (257, 32, 2)
        if self.training:
            #print("training mode")
            freqs_cis = self.freqs_cis[:codes.shape[1]] # (256, 32, 2)
        else:
            '''
            
            model.eval()
            
            if evaluation during training:
                input_pos = torch.arange(256)
                freqs_cis.shape = (256, 32, 2)
                
            if inference(autoregressive, using kv_cache):
                input_pos is a scalar tensor
                freqs_cis.shape = (1, 32, 2)
                
            '''
            #print("input_pos:", input_pos)
            freqs_cis = self.freqs_cis[input_pos]
            #print("freqs_cis.shape:", freqs_cis.shape)
 
        mem_proj = self.input_proj2(mem.to(dtype=self.input_proj2.weight.dtype)) if mem is not None else None  # (B, 256, z_dim=768)

        b, l, c = h.shape # (batch_size, 256, 768)
        
        # if this line of code is added, only memory projection is used, and the input projection is not used.
        # h = torch.zeros_like(h)
        
        #h = h + mem_proj  # (B, 256, z_dim=768)
        
        for layer in self.layers:

            '''
            
            training:
                h.shape: (bs, 256, 768)
                mem_proj.shape: (bs, 256, 768)
                freqs_cis.shape: (256, 32, 2)
                input_pos: None
                mask: None
                
            inference(autoregressive, using kv_cache):
                h.shape: (bs, 1, 768)
                mem_proj.shape: (bs, 1, 768)
                freqs_cis.shape: (1, 32, 2)
                input_pos: (1)
                mask.shape: (bs, 1, 1, 264)
                
            '''
            #zero_mem = torch.zeros_like(mem_proj)  # (B, 256, z_dim=768)

            # print("h.shape:", h.shape)
            # print("mem_proj.shape:", mem_proj.shape if mem_proj is not None else None)
            # print("freqs_cis.shape:", freqs_cis.shape)
            # print("input_pos:", input_pos)
            # print("mask.shape:", mask.shape if mask is not None else None)
             
            h = layer(x=h, mem=mem_proj, freqs_cis=freqs_cis, start_pos=input_pos, mask=mask)
            # h = layer(h, freqs_cis=freqs_cis, start_pos=input_pos, mask=mask)
            
        # h = h + self.cross_attention(self.norm(h), self.norm(mem_proj), freqs_cis=freqs_cis, input_pos=input_pos, mask=mask)
            
        h = self.norm(h)  # (B, L, C)
        predictions = self.output_proj(h)
        
        if targets is not None:
            code_loss = self.mse_loss(predictions, targets)
        
            w = int(np.sqrt(l))

            predictions_reshaped = predictions.reshape(b, w, w, -1).permute(0, 3, 1, 2)
            pred_dec = self.fvae.kl.decode(predictions_reshaped)

            
            gt = targets.reshape(b, w, w, -1).permute(0, 3, 1, 2)
            gt_dec = self.fvae.kl.decode(gt)

            mem = mem.reshape(b, w, w, -1).permute(0, 3, 1, 2)
            mem_dec = self.fvae.vq.decode(mem)
        
            img_loss = self.mse_loss(pred_dec, gt_dec)  # Calculate image reconstruction loss

            if self.use_perceptual_loss:
                ploss = self.perceptual_loss(gt_dec, pred_dec).mean()  # Calculate perceptual loss
                ploss = self.perceptual_weight * ploss  # Add perceptual loss to the
            else:
                ploss = torch.tensor(0.0)
                
            return predictions, code_loss, img_loss, ploss, gt_dec, pred_dec
        
        else:
            return predictions
            
        
    
    
#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_Reg_7B(**kwargs):
    return TransformerReg(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_Reg_3B(**kwargs):
    return TransformerReg(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_Reg_1B(**kwargs):
    return TransformerReg(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_Reg_XXXL(**kwargs):
    return TransformerReg(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_Reg_XXL(**kwargs):
    return TransformerReg(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_Reg_XL(**kwargs):
    return TransformerReg(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_Reg_L(**kwargs):
    return TransformerReg(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_Reg_B(**kwargs):
    return TransformerReg(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

GPT_Reg_models = {
    'GPT-Reg-B': GPT_Reg_B, 'GPT-Reg-L': GPT_Reg_L, 'GPT-Reg-XL': GPT_Reg_XL, 'GPT-Reg-XXL': GPT_Reg_XXL, 'GPT-Reg-XXXL': GPT_Reg_XXXL,
    'GPT-Reg-1B': GPT_Reg_1B, 'GPT-Reg-3B': GPT_Reg_3B, 'GPT-Reg-7B': GPT_Reg_7B, 
}
        
            
            
        