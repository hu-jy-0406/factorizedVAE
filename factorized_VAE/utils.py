from PIL import Image
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import ruamel.yaml as yaml
from tokenizer.tokenizer_image.xqgan_model import VQ_models
from torchvision.utils import save_image

def normalize_image(image, value_range=(-1, 1)):
    """
    Normalize an image tensor to the range [0, 1].
    """
    if isinstance(value_range, tuple) and len(value_range) == 2:
        min_val, max_val = value_range
        image = (image - min_val) / (max_val - min_val)
    else:
        raise ValueError("value_range must be a tuple of (min, max)")
    
    return image.clamp(0, 1)

def process_image(image):
    image = image.squeeze(0).detach().cpu()
    image = image * 0.5 + 0.5  # unnormalize from [-1,1] to [0,1]
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()  # (H, W, C)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    return image
    
def recon_img_with_vae(vae, z, save_path="recon.png"):
    """
    Reconstruct an image from the latent representation z using the VAE.
    """
    with torch.no_grad():
        x_recon = vae.decode(z)
        save_image(x_recon, save_path, normalize=True, value_range=(-1, 1))
        return x_recon
    
def generate_causal_mask(seq_len, device):
    """
    Create an (seq_len, seq_len) causal mask, with True in positions
    that should be masked out (i.e., no access to future positions).
    """
    # Upper-triangular mask of ones => True means "mask out"
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

class TokenDataset(Dataset):
    def __init__(self, npy_path):
        self.tokens = np.load(npy_path)  # shape (N, 8, 8)
        self.tokens = self.tokens.reshape(self.tokens.shape[0], -1)  # shape (N, 64)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return torch.LongTensor(self.tokens[idx])
    
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