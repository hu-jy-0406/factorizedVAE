import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import argparse
import ruamel.yaml as yaml
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
from PIL import Image
from torchvision.datasets import ImageFolder
from tokenizer.tokenizer_image.xqgan_model import VQ_models
from vqvae_simple.encoder import Encoder_Simple
from tokenizer.tokenizer_image.quant import VectorQuantizer2

#export PYTHONPATH=/home/hjy22/repos/ImageFolder:$PYTHONPATH

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

def save_image(image, name=None):
    image = image.squeeze(0).detach().cpu()
    image = image * 0.5 + 0.5  # unnormalize from [-1,1] to [0,1]
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()  # (H, W, C)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(name)

def get_discrete_latent(loader, vq_model):
    latents = []
    for x, y in tqdm(loader, desc="Encoding images"):
        x = x.cuda()
        latent = vq_model.get_quant(x)
        latents.append(latent.detach().cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    print("Latent shape: ", latents.shape)
    # Save latents to a file
    np.save("factorized_VAE/cifar10_train_latents.npy", latents)
    print("Latents saved to factorized_VAE/cifar10_train_latents.npy")

def get_continous_latent(loader, vq_model):
    latents = []
    for x, y in tqdm(loader, desc="Encoding images"):
        x = x.cuda()
        latent = vq_model.encode(x)
        latents.append(latent.detach().cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    print("Latent shape: ", latents.shape)
    # Save latents to a file
    np.save("factorized_VAE/cifar10_val_latents_continous.npy", latents)
    print("Latents saved to factorized_VAE/cifar10_val_latents_continous.npy")

def main(args):
    # set seed
    torch.manual_seed(args.global_seed)
    np.random.seed(args.global_seed)
    torch.cuda.manual_seed_all(args.global_seed)
    
    #prepare dataset
    transform = transforms.Compose([
        #upsample the images from 32*32 to 256*256
        # transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImageFolder("/home/hjy22/repos/CIFAR10/val", transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

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

    #get_continous_latent(loader, vq_model)

    #load the latents
    # latents = np.load("factorized_VAE/cifar10_train_latents.npy")
    # print("Latents shape: ", latents.shape)
    # #change the latents to a tensor
    # batch_size = 100  # or smaller if you still get OOM
    # indices_list = []
    # for i in range(0, latents.shape[0], batch_size):
    #     batch = torch.from_numpy(latents[i:i+batch_size]).cuda()
    #     indices = vq_model.quantize.get_codebook_indices(batch)
    #     indices_list.append(indices.cpu())
    # indices = torch.cat(indices_list, dim=0)
    # print("Indices shape: ", indices.shape)

    # #save the indices to a file
    # np.save("factorized_VAE/cifar10_train_indices.npy", indices.cpu().numpy())
    # print("Indices saved to factorized_VAE/cifar10_train_indices.npy")
    

    #test
    #choose one image randomly from the dataloader
    for i, (x, y) in enumerate(loader):
        if i == 1:
            x = x.cuda()
            break
    #show the image x
    save_image(x, "factorized_VAE/original_image.png")
    #encode
    latent = vq_model.get_quant(x)
    print("latent shape: ", latent.shape)
    #decode
    dec = vq_model.decode(latent)
    print("dec shape: ", dec.shape)
    save_image(dec, "factorized_VAE/reconstructed_image.png")
        
if __name__ == "__main__":
    args = parse_args()
    main(args)