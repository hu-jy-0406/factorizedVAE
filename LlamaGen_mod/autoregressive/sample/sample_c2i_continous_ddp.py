# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.utils import save_image

from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse

from LlamaGen_mod.autoregressive.models import gpt_reg
from LlamaGen_mod.autoregressive.models import gpt
from LlamaGen_mod.tokenizer.tokenizer_image.vq_model import VQ_models
from LlamaGen_mod.autoregressive.models.gpt import GPT_models
from LlamaGen_mod.autoregressive.models.gpt_reg import GPT_Reg_models
from LlamaGen_mod.autoregressive.models.generate import generate, generate_continous_code

from factorized_VAE.my_models.fvae import FVAE


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    fvae = FVAE()
    ckpt_path = "/mnt/disk3/jinyuan/ckpts/fvae/full/fvae_full_3loss+epoch10.pth"
    ckpt = torch.load(ckpt_path)
    fvae.load_state_dict(ckpt["model"])

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    gpt_checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp: # fsdp
        gpt_model_weight = gpt_checkpoint
    elif "model" in gpt_checkpoint:  # ddp
        gpt_model_weight = gpt_checkpoint["model"]
    elif "module" in gpt_checkpoint: # deepspeed
        gpt_model_weight = gpt_checkpoint["module"]
    elif "state_dict" in gpt_checkpoint:
        gpt_model_weight = gpt_checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    # if 'freqs_cis' in model_weight:
    #     model_weight.pop('freqs_cis')
    gpt_model.load_state_dict(gpt_model_weight, strict=False)
    gpt_model.eval()
    del gpt_checkpoint
    
    #create and load got_reg model
    gpt_reg_model = GPT_Reg_models[args.gpt_reg_model](
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type
    ).to(device, dtype=precision)
    print("\n")
    print("args.gpt_reg_ckpt:", args.gpt_reg_ckpt)
    print("\n")
    gpt_reg_checkpoint = torch.load(args.gpt_reg_ckpt, map_location="cpu")
    if args.from_fsdp: # fsdp
        gpt_reg_model_weight = gpt_reg_checkpoint
    elif "model" in gpt_reg_checkpoint:  # ddp
        gpt_reg_model_weight = gpt_reg_checkpoint["model"]
    elif "module" in gpt_reg_checkpoint: # deepspeed
        gpt_reg_model_weight = gpt_reg_checkpoint["module"]
    elif "state_dict" in gpt_reg_checkpoint:
        gpt_reg_model_weight = gpt_reg_checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    gpt_reg_model.load_state_dict(gpt_reg_model_weight, strict=False)
    gpt_reg_model.eval()
    del gpt_reg_checkpoint   
    
    

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
        gpt_reg_model = torch.compile(
            gpt_reg_model,
            mode="reduce-overhead",
            fullgraph=True
        )
    else:
        print(f"no model compile") 

    # Create folder to save samples:
    model_string_name = args.gpt_reg_model.replace("/", "-")
    print(f"model_string_name: {model_string_name}")
    if args.from_fsdp:
        ckpt_string_name = args.gpt_reg_ckpt.split('/')[-2]
    else:
        ckpt_string_name = os.path.basename(args.gpt_reg_ckpt).replace(".pth", "").replace(".pt", "")
    print(f"ckpt_string_name: {ckpt_string_name}")
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-size-{args.image_size_eval}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    print("args.sample_dir:", args.sample_dir)
    print(f"folder_name: {folder_name}")
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    print(f"sample_folder_dir: {sample_folder_dir}")
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)
        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
        # print("c_indices.shape:", c_indices.shape)
        # print("latent_size ** 2:", latent_size ** 2)
        index_sample = generate(
            model=gpt_model, cond=c_indices, max_new_tokens=latent_size ** 2,
            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
            )
        #print("index_sample.shape:", index_sample.shape)
        index_sample = torch.flatten(index_sample) #from (32, 256) to (8192)
        # gen_img = fvae.vq.decode(fvae.vq.quantize.get_codebook_entry(index_sample, shape=(n, 16, 16, 8)))
        # for i in range(n):
        #     index = i * dist.get_world_size() + rank + total
        #     if index < args.num_fid_samples:
        #         save_image(gen_img[i], f"{sample_folder_dir}/{index:06d}.png", normalize=True, value_range=(-1, 1))
        #     else:
        #         break
        #print("index_sample.shape:", index_sample.shape)
        # vq_code = fvae.vq.quantize.get_codebook_entry(index_sample, shape=(n, 16, 16, 8)) # (b, 8, 16, 16)
        # b, c, h, w = vq_code.shape
        # vq_code_dec = fvae.vq.decode(vq_code) # (b, 3, 256, 256)
        quant_code = gpt_reg_model.fvae.vq.quantize.get_codebook_entry(index_sample, shape=(n, 16, 16, 8)) # (b, 8, 16, 16)
        
        print("args.no_mem:", args.no_mem) 
        if args.no_mem==True:
            mem = None
            print("no memory projection, mem is None")
        else:
            print("using memory projection")
            mem = quant_code.permute(0, 2, 3, 1).reshape(n, latent_size ** 2, -1)
            #mem = vq_code.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (b, 256, 8)
        #print("mem.shape:", mem.shape)
        
        
        
        code_sample = generate_continous_code(
            model=gpt_reg_model, cond=c_indices, mem=mem, max_new_codes=latent_size ** 2,
            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval)
        # code_sample.shape = (batch_size, 256, 16)
        
        code_sample = code_sample.reshape(n, latent_size, latent_size, -1).permute(0, 3, 1, 2)  # (b, 16, 16, 16)
        img_sample = fvae.kl.decode(code_sample)  # (b, 3, 256, 256)
        
        mem_code = mem.reshape(n, 16, 16, -1).permute(0, 3, 1, 2)  
        mem_code_dec = fvae.vq.decode(mem_code.to(torch.float32))  # (b, 3, 256, 256)
        
                
        for i in range(n):
            index = i * dist.get_world_size() + rank + total
            if index < args.num_fid_samples:
                save_image(img_sample[i], f"{sample_folder_dir}/{index:06d}.png", normalize=True, value_range=(-1, 1))
                save_image(mem_code_dec[i], f"{sample_folder_dir}/{index:06d}_mem.png", normalize=True, value_range=(-1, 1))
            else:
                break
            
        print(f"Rank {rank} saved {n} samples to {sample_folder_dir}. Total samples saved: {total + n}.")
        
        #TODO#
        #change the shape and decode code_sample# 
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-reg-model", type=str, choices=list(GPT_Reg_models.keys()), default="GPT-Reg-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-reg-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=True)
    parser.add_argument("--no-compile", action='store_false', dest="compile")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--no-mem", action='store_true', default='False', help="whether to use memory projection in the model")
    args = parser.parse_args()
    main(args)