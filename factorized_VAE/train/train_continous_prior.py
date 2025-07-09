import os
import torch
import math
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import numpy as np
from tqdm import tqdm
import argparse
import ruamel.yaml as yaml
import wandb
from tokenizer.tokenizer_image.xqgan_model import VQ_models
from PIL import Image
import tensorflow.compat.v1 as tf
from evaluator import Evaluator
import torch.distributed as dist
from tokenizer.tokenizer_image.lpips import LPIPS

# export MASTER_ADDR=localhost
# export MASTER_PORT=29500
# export WORLD_SIZE=1
# export RANK=0
# export LOCAL_RANK=0
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export PYTHONPATH=/home/hjy22/repos/ImageFolder:$PYTHONPATH
# export PYTHONPATH=/home/renderex/causal_groups/jinyuan.hu/factorizedVAE:$PYTHONPATH
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
os.environ['PYTHONPATH'] = '/home/hjy22/repos/ImageFolder'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

#os.environ["WANDB_MODE"] = "disabled"

if not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_len):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            -math.log(10000.0) * torch.arange(0, embedding_dim, 2).float() / embedding_dim
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x shape: (B, L, d_model)
        returns x + positional embedding
        """
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0).to(x.device)
    
def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--data-path", type=str, default='/mnt/localssd/ImageNet2012/train')
    parser.add_argument("--data-path", type=str, default='/home/hjy22/repos/CIFAR10')
    parser.add_argument("--data-face-path", type=str, default=None, help="face datasets to improve vq model")
    parser.add_argument("--cloud-save-path", type=str, default='output/debug', help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", action='store_true', help="finetune a pre-trained vq model")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--codebook-l2-norm", action='store_true', default=True, help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")
    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-epoch-start", type=int, default=0, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-start", type=int, default=0, help="iteration to start discriminator training and loss")  # autoset
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan'], default='patchgan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge', help="generator loss for gan training")
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--disc_lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", type=str, default='none')
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--disc-weight-decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--vis-every", type=int, default=5000)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--save_best",action='store_true', default=False)
    parser.add_argument("--val_data_path", type=str, default="/mnt/localssd/ImageNet2012/val")
    parser.add_argument("--sample_folder_dir", type=str, default='samples')
    parser.add_argument("--reconstruction_folder_dir", type=str, default='reconstruction')
    parser.add_argument("--v-patch-nums", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16], nargs='+',
                        help="number of patch numbers of each scale")
    parser.add_argument("--enc_type", type=str, default="cnn")
    parser.add_argument("--dec_type", type=str, default="cnn")
    parser.add_argument("--semantic_guide", type=str, default="none")
    parser.add_argument("--detail_guide", type=str, default="none")
    parser.add_argument("--num_latent_tokens", type=int, default=256)
    parser.add_argument("--encoder_model", type=str, default='vit_small_patch14_dinov2.lvd142m',
                        help='encoder model name')
    parser.add_argument("--decoder_model", type=str, default='vit_small_patch14_dinov2.lvd142m',
                        help='encoder model name')
    parser.add_argument("--disc_adaptive_weight", type=bool, default=False)
    parser.add_argument("--abs_pos_embed", type=bool, default=False)
    parser.add_argument("--product_quant", type=int, default=1)
    parser.add_argument("--share_quant_resi", type=int, default=4)
    parser.add_argument("--codebook_drop", type=float, default=0.0)
    parser.add_argument("--half_sem", type=bool, default=False)
    parser.add_argument("--start_drop", type=int, default=1)
    parser.add_argument("--lecam_loss_weight", type=float, default=None)
    parser.add_argument("--sem_loss_weight", type=float, default=0.1)
    parser.add_argument("--detail_loss_weight", type=float, default=0.1)
    parser.add_argument("--enc_tuning_method", type=str, default='full')
    parser.add_argument("--dec_tuning_method", type=str, default='full')
    parser.add_argument("--clip_norm", type=bool, default=False)
    parser.add_argument("--sem_loss_scale", type=float, default=1.0)
    parser.add_argument("--detail_loss_scale", type=float, default=1.0)
    parser.add_argument("--config", type=str, default=None, help="config file path")
    parser.add_argument("--norm_type", type=str, default='bn')
    parser.add_argument("--aug_prob", type=float, default=1.0)
    parser.add_argument("--aug_fade_steps", type=int, default=0)
    parser.add_argument("--disc_reinit", type=int, default=0)
    parser.add_argument("--debug_disc", type=bool, default=False)
    parser.add_argument("--guide_type_1", type=str, default='class', choices=["patch", "class"])
    parser.add_argument("--guide_type_2", type=str, default='class', choices=["patch", "class"])
    parser.add_argument("--lfq", action='store_true', default=False, help="if use LFQ")

    parser.add_argument("--end-ratio", type=float, default=0.5)
    parser.add_argument("--anneal-start", type=int, default=100)
    parser.add_argument("--anneal-end", type=int, default=200)
    
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--delta", type=int, default=100)
    parser.add_argument("--use_diffloss", action='store_true', default=False, help="if use diffloss")
    parser.add_argument("--use_perceptual_loss", action='store_true', default=False, help="if use perceptual loss")
    parser.add_argument("--use_disc_loss", action='store_true', default=False, help="if use discriminator loss")
    parser.add_argument("--use_lecam_loss", action='store_true', default=False, help="if use lecam loss")
    parser.add_argument("--train_stage", type=str, default='full', choices=['full', 'diff_only', 'dec_only'], help="train stage")
    parser.add_argument("--use_latent_perturbation", default=False, help="if use latent perturbation")
    parser.add_argument("--wandb_project", type=str, default="xqgan")

    #for sampling and FID calculation
    # parser.add_argument("--sample-dir", type=str, default="samples", help="Directory to save generated samples")
    # parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples to generate for FID calculation")
    # parser.add_argument("--seed", type=int, default=0, help="Random seed")
    # parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    # parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling parameter (0 for no top-k)")
    # parser.add_argument("--top-p", type=float, default=0.0, help="Top-p sampling parameter (0 for no top-p)")

    args = parser.parse_args()

    if args.config is None:
        # if no config file is provided, use the default config file
        args.config = 'configs/VQ-8192.yaml'
        print(f"No config file provided, using default config: {args.config}")

    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    else:
        raise ValueError("Please provide a config file path with --config argument.")

        # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()

    if args.train_stage == 'diff_only' or args.train_stage == 'dec_only':
        assert args.vq_ckpt is not None, "Please provide a checkpoint to resume training from."

    if args.train_stage == 'full' or args.train_stage == 'dec_only':
        args.use_diffloss = False
    elif args.train_stage == 'diff_only':
        args.use_diffloss = True

    return args

def process_image(image):
    image = image.squeeze(0).detach().cpu()
    image = image * 0.5 + 0.5  # unnormalize from [-1,1] to [0,1]
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()  # (H, W, C)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    return image

def save_image(image, name=None):
    image = process_image(image)
    image.save(name)

class TokenDataset(Dataset):
    def __init__(self, npy_path):
        self.tokens = np.load(npy_path)  # shape (N, 8, 8)
        self.tokens = self.tokens.reshape(self.tokens.shape[0], -1)  # shape (N, 64)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return torch.LongTensor(self.tokens[idx])
    
class LatentDataset(Dataset):
    def __init__(self, npy_path):
        self.latents = np.load(npy_path)  # shape (N, 32, 8, 8)
        self.latents = torch.FloatTensor(self.latents).permute(0, 2, 3, 1)  # shape (N, 8, 8, 32)
        #reshape to (N, 64, 32)
        N, H, W, C = self.latents.shape
        self.latents = self.latents.reshape(N, H * W, C)  # shape (N, 64, 32)
    def __len__(self):
        return self.latents.shape[0]
    def __getitem__(self, idx):
        return self.latents[idx]

def generate_causal_mask(seq_len, device):
    """
    Create an (seq_len, seq_len) causal mask, with True in positions
    that should be masked out (i.e., no access to future positions).
    """
    # Upper-triangular mask of ones => True means "mask out"
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

class DiscretePrior(nn.Module):
    """
    Autoregressive Transformer using nn.TransformerDecoder.
    Predict token i given [CLS] and tokens < i.
    """
    def __init__(self, vocab_size, seq_len=64, d_model=512, nhead=8, num_layers=8, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Embeddings + sinusoidal positional embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_len=seq_len)

        # TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        input: <s>, x1, x2, ...,xL-1
        output: x1',x2',x3',...,xL'
        """
        B, L = x.shape
        # Prepend <s> token (assume <s> ID = 0)
        start_token = torch.zeros((B, 1), dtype=torch.long, device=x.device)
        x = torch.cat([start_token, x], dim=1)  # => (B, L)
        x = x[:, :L]

        # Token embedding => shape (B, L, d_model)
        embeds = self.token_emb(x)

        # Add sinusoidal position embeddings => shape (B, L, d_model)
        h = self.pos_emb(embeds)

        # Create causal mask => shape (L, L)
        causal_mask = generate_causal_mask(L, x.device)

        # Dummy memory => shape (B, 1, d_model)
        memory = torch.zeros(B, 1, self.d_model, device=x.device)

        out = self.decoder(tgt=h, memory=memory, tgt_mask=causal_mask)
        out = self.ln(out)  # => (B, L, d_model)
        logits = self.head(out)  # => (B, L, vocab_size)

        return logits

    @torch.no_grad()
    def get_next_token(self, x):
        """
        Get the next token prediction for input x.
        x shape: (B, L)
        returns: next token logits shape (B, vocab_size)
        """
        B, L = x.shape
        memory = torch.zeros(B, 1, self.d_model, device=device)
        embs = self.token_emb(x)  # shape (B, L, d_model)
        h = self.pos_emb(embs)  # shape (B, L, d_model)
        out = self.decoder(tgt=h, memory=memory, tgt_mask=None)
        out = self.ln(out)  # shape (B, L, d_model)
        logits = self.head(out)  # shape (B, L, vocab_size)
        last_logits = logits[:, -1, :]
        p = last_logits.softmax(dim=-1)  # Apply softmax to get probabilities
        #sample from the distribution p
        next_token = torch.multinomial(p, num_samples=1)
        return next_token  # shape (B, 1)


    @torch.no_grad()
    def generate(self, num_samples):
        device = next(model.parameters()).device

        token_list = []
        start_token = torch.zeros((num_samples,1)).long().to(device) # Start with <s> token
        token_list.append(start_token)  # Append <s> token ID (assumed to be 0)

        for i in range(self.seq_len):
            tokens = torch.cat(token_list, dim=1).to(device)  # shape (B, i+1)
            next_token = self.get_next_token(tokens)
            token_list.append(next_token) # Update the next token in the sequence

        tokens = torch.cat(token_list, dim=1)  # shape (B, seq_len+1)
        tokens = tokens[:,1:]

        return tokens
    
class ContinousPrior(nn.Module):
    """
    Autoregressive Transformer using nn.TransformerDecoder.
    Predict token i given [CLS] and tokens < i.
    """
    def __init__(self, vq_model, z_dim=32, seq_len=64, d_model=512, nhead=8, num_layers=8, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.z_dim = z_dim
        for param in vq_model.parameters():
            param.requires_grad = False
        self.vq_model= vq_model  # Use the quantizer from the VQ model
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
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, z_dim)
        
        self.temp = 0

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
        #x = x[:, :L, :]
        
        # For each position i, predict x_i using context x_1, ..., x_{i-1} and memory \hat{x_i}
        '''
        inputs: x1, x2, ..., xi-1
        memory: \hat{x_i} (specific memory for position i)
        outputs: x_i (prediction for position i)
        '''
        for i in range(L):
            context = x[:, :i+1, :]
            
            # Project context to model dimension
            context_proj = self.input_proj(context)  # (B, i+1, d_model)
            
            # Add positional embeddings
            context_pos = self.pos_emb(context_proj)  # (B, i+1, d_model)
            
            # Get memory for position i
            mem_i = x[:, i+1:i+2, :].unsqueeze(2).permute(0,3,1,2)  # (B, 1, z_dim)
            mem_i,_,_,_,_ = self.vq_model.quantize.forward(mem_i, ret_usages=True, dropout=None) # Quantize memory token
            mem_i = mem_i.permute(0,2,3,1)
            mem_i = mem_i.squeeze(2).detach()
            mem_i_proj = self.input_proj(mem_i)  # (B, 1, d_model)
            zero_memory = torch.zeros_like(mem_i_proj)  # shape (B, 1, d_model)
            
            # Run decoder
            # No need for causal mask since we're only providing valid context
            out = self.decoder(
                tgt=context_pos,
                memory=mem_i_proj,  # Memory for position i
                tgt_mask=None  # Context is already properly limited
            )
            
            # Get prediction for position i
            pred = self.output_proj(self.ln(out))  # (B, C)
            pred_i = pred[:, -1:, :]  # Get the last prediction (for position i)
            # Store prediction
            #predictions[:, i, :] = pred_i
            predictions.append(pred_i)  # Append the prediction for position i
            #predictions.append(x[:, i+1:i+2, :])

        predictions = torch.cat(predictions, dim=1)  # shape (B, L, z_dim)
        mse_loss = nn.MSELoss()(predictions, x_origin)  # Calculate loss against original input
        #print(f"Loss: {loss.item()}")

        #add perceptual loss
        # p_loss = self.perceptual_loss(predictions.permute(0,2,1).reshape(B, C, 8, 8), x_origin.permute(0,2,1).reshape(B, C, 8, 8))
        # loss += self.perceptual_weight * p_loss.mean()  # Add perceptual loss to the total loss
        

        samples = []

        B, L, C = predictions.shape
        for i in range(B):
            latents_i = predictions[i].reshape(8, 8, C)
            latents_i = torch.tensor(latents_i).unsqueeze(0).permute(0,3,1,2) # shape (1, C, 8, 8)
            #print(f"latents_i shape: {latents_i.shape}, tokens shape: {tokens[i].shape}")
            latents_i,_,_,_,_ = vq_model.quantize(latents_i, ret_usages=True, dropout=None)# Quantize the latents
            img_i_dec = vq_model.decode(latents_i).to('cpu')  # shape (1, 3, 256, 256)
            img_i = process_image(img_i_dec)  # Convert to PIL Image
            samples.append(img_i)  # Append the image to samples
        #img_i.save(f"factorized_VAE/ContinousPrior/dec.png")  # Save the image

        for i in range(B):
            gt = x_origin[i].reshape(8, 8, C)
            gt = torch.tensor(gt).unsqueeze(0).permute(0,3,1,2)
            gt,_,_,_,_ = self.vq_model.quantize(gt, ret_usages=True, dropout=None)
            gt_dec = self.vq_model.decode(gt).to('cpu')
            # print(self.temp == gt_dec)
            # self.temp = gt_dec
            gt_img = process_image(gt_dec)
            samples.append(gt_img)

        #show all the sampled 32*32 PIL images in one image with 1*2 grid
        num_col = int(len(samples)/2)
        if len(samples) > 0:
            grid_img = Image.new('RGB', (32 * num_col, 32*2))
            for i, img in enumerate(samples):
                grid_img.paste(img, (i % num_col * 32, i // num_col * 32))
            grid_img.save("factorized_VAE/ContinousPrior/dec_epoch.png")

        gt_dec_detach = gt_dec.clone().to(device).detach()  # Detach the ground truth image for perceptual loss

        if self.use_perceptual_loss:
            ploss = self.perceptual_loss(gt_dec_detach, img_i_dec.to(device)).mean()  # Calculate perceptual loss
            ploss = self.perceptual_weight * ploss  # Add perceptual loss to the
        else:
            ploss = torch.tensor(0.0, device=device)      


        return predictions, mse_loss, ploss, grid_img

    @torch.no_grad()
    def get_next_latent(self, latents, next_token):
        B, L, C = latents.shape
        # #codebook_lookup
        memory = self.vq_model.quantize.codebook_lookup_flat(next_token)  # shape (B, 1, z_dim)
        memory_proj= self.input_proj(memory)  # Project to d_model
        # zero_memory = torch.zeros_like(memory_proj)  # shape (B, 1, d_model)  
        h = self.pos_emb((self.input_proj(latents)))  # shape (B, L, d_model)

        # h = self.input_proj(latents)  # shape (B, L, d_model)
        # #h = torch.zeros_like(h)  # Reset h to zeros for the next token prediction

        out = self.decoder(tgt=h, memory=memory_proj, tgt_mask=None)  # shape (B, L, d_model)
        logits = self.output_proj(self.ln(out))  # shape (B, L, d_model)

        # logits = self.output_proj(h)  # shape (B, L, z_dim)
        last_logits = logits[:, -1, :]  # Get the last prediction (for next token)

        #last_logits = latents[:, -1, :]
        #last_logits = torch.rand_like(last_logits)

        return last_logits.unsqueeze(1)  # shape (B, 1, z_dim)
        #return memory


        
    ##TODO##
    @torch.no_grad()
    def generate(self, discrete_prior, num_samples):
        device = next(model.parameters()).device
        
        token_list = []
        latent_list = []

        start_token = torch.zeros((num_samples,1)).long().to(device)
        start_latent = torch.zeros((num_samples, 1, self.z_dim), device=device)  # Start with zero vector

        token_list.append(start_token)  # Append start token ID (assumed to be 0)
        latent_list.append(start_latent)  # Append start token

        for i in range(self.seq_len):
            tokens = torch.cat(token_list, dim=1).to(device)
            latents = torch.cat(latent_list, dim=1).to(device)
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
    

def train(model, discrete_prior, resume_path=None):
    wandb.login(key="3761ca3994faa9fea7e291585ce72a0ed49562a0")
    logger = wandb.init(project="ContinousPrior_Planes")
    # logger.define_metric("loss", step_metric="epoch")
    # logger.define_metric("acc_val", step_metric="epoch")

    # Hyperparameters
    npy_path = "factorized_VAE/cifar10_val_latents_continous.npy"
    batch_size = 16
    lr = 1e-4
    epochs = 500
    vis_every = 100
    

    dataset = LatentDataset(npy_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    #use only one data from dataset to construct a new loader
    test_dataset = [dataset[1], dataset[2], dataset[9], dataset[11], dataset[12], dataset[16], dataset[17]] #1,2,9,11,12,16,17
    test_loader = DataLoader(test_dataset, batch_size=7, shuffle=False, num_workers=1, pin_memory=True)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.MSELoss()  # Use MSELoss for continuous prior

    start_epoch = 0
    global_step = 0

    # If you have a resume checkpoint, load it here
      # update this if needed
    if resume_path and os.path.isfile(resume_path):
        print("Loading resume checkpoint...")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        print(f"Resumed from epoch {start_epoch} at global step {global_step}.")

    model.train()
    for epoch in range(start_epoch, epochs):
        # data = []
        # data.append(dataset[0].unsqueeze(0))
        pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        total_loss = 0
        for batch in pbar:
            # batch shape: (B, seq_len=64)
            batch = batch.to(device)
            
            # Forward pass => logits (B, L, vocab_size)
            logits, mse_loss, ploss, grid_img = model(batch)
            loss = mse_loss + ploss  # Combine losses
            
            logger.log({"batch_totalloss": loss.item()}, step=global_step)
            logger.log({"batch_ploss": ploss.item()}, step=global_step)
            logger.log({"batch_mseloss": mse_loss.item()}, step=global_step)
            global_step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataset)
        logger.log({"epoch_total_loss": avg_loss, "epoch": epoch+1},step=global_step)
        #print(f"Epoch {epoch+1} average loss: {avg_loss.item():.4f}")

        # Save checkpoint
        if (epoch+1) % 500 == 0 or (epoch+1) == epochs:
            ckpt_state = {
                "epoch": epoch+1,
                "global_step": global_step+1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            ckpt_path = f"factorized_VAE/ContinousPrior/ContinousPrior_single_disweight_ploss_epoch{epoch+1}.pth"
            torch.save(ckpt_state, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

        if (epoch+1) % vis_every == 0 or (epoch+1) == epochs:
            # Sample and visualize
            # grid_img = sample(model, discrete_prior, num_samples=20)
            # if grid_img is not None:
            #     logger.log({"sampled_images": wandb.Image(grid_img)}, step=global_step)
            logger.log({"grid_image": wandb.Image(grid_img)}, step=global_step)

        #torch.save(model.state_dict(), f"factorized_VAE/token_transformer_decoder_epoch{epoch+1}.pth")

def sample(model, discrete_prior, num_samples=20):
    """
    Generate samples from the model.
    """
    model.eval()
    samples = []

    batch_size = 20
    num_batches = math.ceil(num_samples / batch_size)
    

    for j in range(num_batches):
        batch_size = min(batch_size, num_samples - len(samples))

        tokens, latents = model.generate(discrete_prior, num_samples=batch_size)
        # tokens shape: (batch_size, seq_len=64)
        # latents shape: (batch_size, seq_len=64, z_dim=32)

        #-----check whether the tokens are correct-----
        #get latents from tokens
        # tokens = tokens.reshape(tokens.shape[0], 8, 8)  # Reshape to (batch_size, 8, 8)
        # latents = vq_model.quantize.codebook_lookup(tokens)  # shape (batch_size, 8, 8, z_dim)
        # latents = latents.permute(0, 2, 3, 1)  # Change to (batch_size, 8, 8, z_dim)
        # latents = latents.reshape(latents.shape[0], 64, -1)  # Reshape to (batch_size, 64, z_dim)
        #---------------------------------------------

        for i in range(latents.shape[0]):
            B, L, C = latents.shape
            latents_i = latents[i].reshape(8, 8, C)
            latents_i = torch.tensor(latents_i).unsqueeze(0).permute(0,3,1,2) # shape (1, C, 8, 8)

            #print(f"latents_i shape: {latents_i.shape}, tokens shape: {tokens[i].shape}")
            
            #latents_i,_,_,_,_ = vq_model.quantize(latents_i, ret_usages=True, dropout=None)# Quantize the latents

            img_i = vq_model.decode(latents_i).to('cpu')  # shape (1, 3, 256, 256)
            img_i = process_image(img_i)  # Convert to PIL Image
            samples.append(img_i)
    
    print(f"Generated {len(samples)} samples")

    # show all the sampled 32*32 PIL images in one image with 2x5 grid
    if len(samples) > 0:
        grid_img = Image.new('RGB', (32 * 5, 32 * 4))
        for i, img in enumerate(samples):
            grid_img.paste(img, (i % 5 * 32, i // 5 * 32))
        #grid_img.show()  # Display the grid of images
        grid_img.save("factorized_VAE/samples.png")  # Save the grid image
    else:
        grid_img = None
        print("No samples generated.")
    #show grid image in wandb

    return grid_img

    # generated_tokens = model.generate().cpu()  # shape (1, 64)
    # print("shape of generated tokens:", generated_tokens.shape)
    # print("Generated tokens:", generated_tokens)
    # #generated_tokens = model.autoregressive_inference(seq_len=64, start_token_id=0, gt=x_reshaped)
    # gen_tokens = np.array(generated_tokens).reshape(8, 8)
    # #change 8*8 array to 1*8*8 tensor
    # gen_tokens = torch.tensor(gen_tokens).unsqueeze(0)  # shape (1, 8, 8)

    # gen_latents = vq_model.quantize.codebook_lookup(gen_tokens)  # shape (1, 8, 8, codebook_embed_dim)
    # gen_latents = gen_latents.squeeze(0)  # shape (8, 8, codebook_embed_dim)
    # gen_img = vq_model.decode(gen_latents)  # shape (1, 3, 256, 256)
    # save_image(gen_img, "factorized_VAE/generated_image.png")
    


def evaluation_FID(model, val_data_path):
    #prepare dataset
    transform = transforms.Compose([
        #upsample the images from 32*32 to 256*256
        # transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(val_data_path, transform=transform)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    num_samples = len(dataset)
    print(f"Number of samples in validation dataset: {num_samples}")

    batch_size = 128  # Adjust batch size as needed

    #sample images from the model
    model.eval()
    generated_samples = []
    total = 0
    with torch.no_grad():
        for _ in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
            current_batch_size = min(batch_size, num_samples - total)
            
            # Generate tokens
            tokens = model.generate(num_samples=current_batch_size)
            
            # Process tokens to images in batch
            batch_tokens = []
            for i in range(tokens.shape[0]):
                tokens_i = tokens[i].reshape(8, 8)
                batch_tokens.append(tokens_i)
                
            # Convert to tensor and process in batch
            batch_tokens = torch.stack(batch_tokens).to(device)  # shape (batch_size, 8, 8)
            latents = vq_model.quantize.codebook_lookup(batch_tokens)  # shape (batch_size, 8, 8, codebook_embed_dim)
            images = vq_model.decode(latents)  # shape (batch_size, 3, 32, 32)
            
            # Process for FID calculation (following xqgan_train.py format)
            images = torch.clamp(127.5 * images + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
            generated_samples.append(images.cpu().numpy())
            
            total += current_batch_size
            if total >= num_samples:
                break
    
    # Concatenate all generated samples
    generated_samples = np.concatenate(generated_samples, axis=0)
    print(f"Generated {generated_samples.shape[0]} samples for FID calculation")   

    

    gt = []
    for x, _ in tqdm(val_loader, desc="Loading validation data"):
        # Process images for FID calculation
        x = torch.clamp(255 * x, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
        gt.append(x.numpy())
    gt = np.concatenate(gt, axis=0)

    fid, is_score = calculate_fid(generated_samples, gt)
    print(f"Generation FID: {fid:.4f}")
    print(f"Inception Score: {is_score:.4f}")

def load_validation_data(args):
    """Load CIFAR-10 validation data for reference FID calculation"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    val_dataset = datasets.ImageFolder(args.val_data_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    gt = []
    for x, _ in tqdm(val_loader, desc="Loading validation data"):
        # Process images for FID calculation
        x = torch.clamp(255 * x, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
        gt.append(x.numpy())
    
    gt = np.concatenate(gt, axis=0)
    print(f"Loaded {gt.shape[0]} validation images")
    return gt

def calculate_fid(generated_samples, real_samples):
    """Calculate FID between generated and real samples"""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    
    evaluator = Evaluator(tf.Session(config=config), batch_size=128)
    evaluator.warmup()
    
    print("Computing reference batch activations...")
    ref_acts = evaluator.read_activations(real_samples)
    print("Computing reference batch statistics...")
    ref_stats, _ = evaluator.read_statistics(real_samples, ref_acts)
    
    print("Computing generated sample activations...")
    sample_acts = evaluator.read_activations(generated_samples)
    print("Computing generated sample statistics...")
    sample_stats, _ = evaluator.read_statistics(generated_samples, sample_acts)
    
    fid = sample_stats.frechet_distance(ref_stats)
    # print(f"Generation FID: {fid:.4f}")
    
    # Calculate Inception Score
    is_score = evaluator.compute_inception_score(sample_acts[0])
    # print(f"Inception Score: {is_score:.4f}")
    
    return fid, is_score

def load_discrete_backbone(continous_prior_model, discrete_prior_ckpt_path):
    """
    Load the TransformerDecoder weights from a DiscretePrior checkpoint into a ContinousPrior model.

    Args:
        continous_prior_model: The ContinousPrior model to initialize.
        discrete_prior_ckpt_path: Path to the DiscretePrior checkpoint.
    """
    # Load the DiscretePrior checkpoint
    ckpt = torch.load(discrete_prior_ckpt_path, map_location="cpu")
    discrete_prior_state_dict = ckpt["model"]

    print(f"Loading TransformerDecoder weights from {discrete_prior_ckpt_path} into ContinousPrior model...")

    # Extract the TransformerDecoder weights from the DiscretePrior checkpoint
    transformer_decoder_keys = [
        key for key in discrete_prior_state_dict.keys() if key.startswith("decoder.")
    ]

    # Map the keys to the ContinousPrior model
    continous_prior_state_dict = continous_prior_model.state_dict()
    for key in transformer_decoder_keys:
        # Replace "decoder." with "decoder." to match the ContinousPrior model's key structure
        #new_key = key.replace("decoder.", "decoder.")
        if key in continous_prior_state_dict:
            continous_prior_state_dict[key] = discrete_prior_state_dict[key]
        else:
            print(f"Warning: Key {key} not found in ContinousPrior model.")

    # Load the updated state dict into the ContinousPrior model
    continous_prior_model.load_state_dict(continous_prior_state_dict, strict=False)
    print("TransformerDecoder weights successfully loaded.")

if __name__ == "__main__":
    args = parse_args()
    #load vq_model
    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
        v_patch_nums=args.v_patch_nums,
        enc_type=args.enc_type,
        encoder_model=args.encoder_model,
        dec_type=args.dec_type,
        decoder_model=args.decoder_model,
        semantic_guide=args.semantic_guide,
        detail_guide=args.detail_guide,
        num_latent_tokens=args.num_latent_tokens,
        abs_pos_embed=args.abs_pos_embed,
        share_quant_resi=args.share_quant_resi,
        product_quant=args.product_quant,
        codebook_drop=args.codebook_drop,
        half_sem=args.half_sem,
        start_drop=args.start_drop,
        sem_loss_weight=args.sem_loss_weight,
        detail_loss_weight=args.detail_loss_weight,
        clip_norm=args.clip_norm,
        sem_loss_scale=args.sem_loss_scale,
        detail_loss_scale=args.detail_loss_scale,
        guide_type_1=args.guide_type_1,
        guide_type_2=args.guide_type_2,
        lfq=args.lfq,
        use_diffloss=args.use_diffloss,
        use_latent_perturbation=args.use_latent_perturbation,
        train_stage=args.train_stage
    )

    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    state_dict = checkpoint["model"]

    # 过滤掉 diffloss 相关的参数
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('diffloss')}
    # 加载到新模型（假设 vq_model 是新实例化的模型，且 diffloss 已经是新的）
    missing, unexpected = vq_model.load_state_dict(filtered_state_dict, strict=False)
    vq_model.eval().cuda()


    vocab_size = 8192  # set this to match your codebook size
    seq_len = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    discrete_prior = DiscretePrior(vocab_size, seq_len=seq_len, d_model=512, nhead=8, num_layers=8).to(device)
    discrete_resume_path = "factorized_VAE/DiscretePrior_val_epoch2000.pth"
    ckpt = torch.load(discrete_resume_path, map_location=device)
    discrete_prior.load_state_dict(ckpt["model"])

    model = ContinousPrior(vq_model=vq_model, z_dim=32, seq_len=seq_len, d_model=512, nhead=8, num_layers=8).to(device)

    load_discrete_backbone(model, discrete_resume_path)

    #resume_path = "factorized_VAE/ContinousPrior/ContinousPrior_val_epoch500.pth"
    resume_path = None

    #sample(model, discrete_prior, num_samples=20)

    train(model, discrete_prior, resume_path=resume_path)
    # ckpt = torch.load(resume_path, map_location=device)
    # model.load_state_dict(ckpt["model"])

    #model.generate(num_samples=10)

    #sample(model, discrete_prior, num_samples=20)
    
    #evaluation_FID(model,"/home/hjy22/repos/CIFAR10/val")

    print("Done!")


