from builtins import print
from calendar import c
from doctest import OutputChecker
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, MSELoss
from LlamaGen_mod.autoregressive.models.gpt import TransformerBlock, Transformer, ModelArgs, CrossAttention
from LlamaGen_mod.autoregressive.models.gpt_reg import TransformerReg
from factorized_VAE.my_models.fvae import FVAE
from tokenizer.tokenizer_image.lpips import LPIPS
from typing import Optional
import numpy as np
from tqdm import tqdm

# for debug
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

                
class TransformerFn(TransformerReg):
    def __init__(self, config: ModelArgs):
        super().__init__(config)

        del self.input_proj
        del self.cls_embedding
        
    def forward(
            self,
            input: torch.Tensor,
            targets: Optional[torch.Tensor]=None,
    ):
        freqs_cis = self.freqs_cis[:input.shape[1]].to(input.device)
        mem_proj = self.input_proj2(input.to(dtype=self.input_proj2.weight.dtype))
        h = mem_proj
        # print("h.device:", h.device)
        # print("freqs_cis.device:", freqs_cis.device)
        for layer in self.layers:
            h = layer(x=h, freqs_cis=freqs_cis, start_pos=None)

        h = self.norm(h)
        logits = self.output_proj(h)

        if targets is None:
            return logits
        else:
            loss = F.mse_loss(logits, targets.to(dtype=logits.dtype))
            return logits, loss

#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_Fn_7B(**kwargs):
    return TransformerFn(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_Fn_3B(**kwargs):
    return TransformerFn(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_Fn_1B(**kwargs):
    return TransformerFn(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_Fn_XXXL(**kwargs):
    return TransformerFn(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_Fn_XXL(**kwargs):
    return TransformerFn(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_Fn_XL(**kwargs):
    return TransformerFn(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_Fn_L(**kwargs):
    return TransformerFn(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_Fn_B(**kwargs):
    return TransformerFn(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

GPT_Fn_models = {
    'GPT-Fn-B': GPT_Fn_B, 'GPT-Fn-L': GPT_Fn_L, 'GPT-Fn-XL': GPT_Fn_XL, 'GPT-Fn-XXL': GPT_Fn_XXL, 'GPT-Fn-XXXL': GPT_Fn_XXXL,
    'GPT-Fn-1B': GPT_Fn_1B, 'GPT-Fn-3B': GPT_Fn_3B, 'GPT-Fn-7B': GPT_Fn_7B, 
}
        
            
            
        