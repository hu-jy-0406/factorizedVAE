# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
from tracemalloc import start
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image
from glob import glob
from copy import deepcopy
import os
import time
import inspect
import argparse
import wandb
from datetime import datetime
import numpy as np

from LlamaGen_mod.utils.logger import create_logger
from LlamaGen_mod.utils.distributed import init_distributed_mode
from LlamaGen_mod.utils.ema import update_ema, requires_grad
from LlamaGen_mod.dataset.build import build_dataset
from LlamaGen_mod.autoregressive.models.gpt import GPT_models
from LlamaGen_mod.autoregressive.models.generate import generate

from factorized_VAE.my_models.fvae import FVAE

import torch._dynamo
torch._dynamo.config.suppress_errors = True #important! This prevents torch._dynamo from raising errors when it encounters unsupported operations, which can happen with certain model architectures or configurations.


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer



#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    
    # Setup wandb logger
    if rank == 0:
        wandb.login(key="3761ca3994faa9fea7e291585ce72a0ed49562a0")
        wandb_logger = wandb.init(project="LlamaGen",
                            name=str(datetime.now().strftime('%Y.%m.%d-%H.%M.%S'))+f"rank-{rank}")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-")  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    fvae = FVAE()
    ckpt_path = "/mnt/disk3/jinyuan/ckpts/fvae/full/fvae_full_3loss+epoch10.pth"
    ckpt = torch.load(ckpt_path)
    fvae.load_state_dict(ckpt["model"])    
    
    # Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # Setup data:
    # ----------- setup train data -----------
    train_dataset = build_dataset("train", args)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    flip_info = 'with' if train_dataset.flip else 'without'
    aug_info = 10 if 'ten_crop' in train_dataset.feature_dir else 1
    aug_info = 2 * aug_info if train_dataset.aug_feature_dir is not None else aug_info
    logger.info(f"Train dataset contains {len(train_dataset):,} images ({args.train_code_path}) "
                f"{flip_info} flip augmentation and {aug_info} crop augmentation")
    # ----------- setup data for visualization fitting result ------------ #
    vis_dataset = Subset(train_dataset, list(range(args.vis_num)))
    vis_loader = DataLoader(vis_dataset, batch_size=args.vis_num, shuffle=False, pin_memory=True)
    # ----------- setup val data -----------
    val_dataset = build_dataset("val", args)
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    flip_info = 'with' if val_dataset.flip else 'without'
    aug_info = 10 if 'ten_crop' in val_dataset.feature_dir else 1
    aug_info = 2 * aug_info if val_dataset.aug_feature_dir is not None else aug_info
    logger.info(f"Val dataset contains {len(val_dataset):,} images ({args.val_code_path}) "
                f"{flip_info} flip augmentation and {aug_info} crop augmentation")
    
    # Prepare models for training:
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "steps" in checkpoint:
            train_steps = checkpoint["steps"]
        train_steps = checkpoint["steps"] if "steps" in checkpoint else 0
        start_epoch = checkpoint["epochs"].epochs if "epochs" in checkpoint else 0
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
    else:
        train_steps = 0
        start_epoch = 0
        
    logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    
    if args.ema:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        
    
    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(-1)
            assert z_indices.shape[0] == c_indices.shape[0]
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                _, loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices)
            #c_indices.shape = ([60]),z_indices.shape = ([60, 256]), idx.shape = ([60, 255])
            # backward pass, with gradient scaling if training in fp16         
            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            if args.ema:
                update_ema(ema, model.module._orig_mod if not args.no_compile else model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if rank == 0:
                    wandb_logger.log({"train_loss": avg_loss, "steps_per_sec": steps_per_sec, "epoch": epoch}, step=train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()
            
            # Evaluation    
            if train_steps % args.val_every == 0 and train_steps > 0:
                logger.info(f"Begining evaluation at step{train_steps}, epoch{epoch}")
                model.eval()
                val_steps = 0
                total_val_loss = 0.0
                with torch.no_grad():
                    for x, y in val_loader:
                        x = x.to(device, non_blocking=True)#(256, 1, 1, 256)
                        y = y.to(device, non_blocking=True)#(256, 1)
                        z_indices = x.reshape(x.shape[0], -1)# ([256, 256])
                        c_indices = y.reshape(-1)# ([256])
                        assert z_indices.shape[0] == c_indices.shape[0]
                        with torch.cuda.amp.autocast(dtype=ptdtype):  
                            _, loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], input_pos=torch.arange(256), targets=z_indices)
                        total_val_loss += loss.item()
                        val_steps += 1
                    print(f"total_val_loss = {total_val_loss}, val_steps = {val_steps}")
                avg_val_loss = torch.tensor(total_val_loss / val_steps, device=device)
                dist.all_reduce(avg_val_loss, op=dist.ReduceOp.SUM)
                avg_val_loss = avg_val_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Val Loss: {avg_val_loss:.4f}")
                if rank == 0:
                    wandb_logger.log({"total_val_loss": total_val_loss}, step=train_steps)
                    wandb_logger.log({"val_loss": avg_val_loss}, step=train_steps)
                model.train()
                
            # Visualize fitting result and generation result
            if train_steps % args.vis_every == 0 and train_steps > 0:
                logger.info(f"Begining visualization at step{train_steps}, epoch{epoch}")
                model.eval()
                with torch.no_grad():
                    # Visualize fitting results
                    for x, y in vis_loader:
                        x = x.to(device, non_blocking=True)#(b, 1, 1, 256)
                        y = y.to(device, non_blocking=True)#(b, 1)
                        z_indices = x.reshape(x.shape[0], -1)# (b, 256)
                        c_indices = y.reshape(-1)# ([256])
                        assert z_indices.shape[0] == c_indices.shape[0]
                        with torch.cuda.amp.autocast(dtype=ptdtype):  
                            logits, _ = model(cond_idx=c_indices, idx=z_indices[:,:-1], input_pos=torch.arange(256), targets=z_indices)
                            # logits.shape = (b, 256, 16384)
                            b, _, _ = logits.shape
                            probs = F.softmax(logits, dim=-1)
                            indices = torch.argmax(probs, dim=-1)
                            indices = torch.flatten(indices)
                            z_q = fvae.vq.quantize.get_codebook_entry(indices, shape=(b, 16, 16, 8))
                            recon_img = fvae.vq.decode(z_q)# (b, 3, 256, 256)
                        gt_indices = torch.flatten(x)
                        gt_img = fvae.vq.decode(fvae.vq.quantize.get_codebook_entry(gt_indices, shape=(b, 16, 16, 8)))
                        recon_grid = make_grid(recon_img, nrow=args.vis_num, normalize=True, value_range=(-1, 1))
                        gt_grid = make_grid(gt_img, nrow=args.vis_num, normalize=True, value_range=(-1, 1))
                        combined_grid = torch.cat((gt_grid, recon_grid), dim=1)
                        combined_image = combined_grid.permute(1, 2, 0).cpu().numpy()
                        combined_image = (combined_image * 255).astype(np.uint8)
                        if rank == 0:
                            wandb_logger.log({"Reconstruction Comparison": wandb.Image(combined_image)}, step=train_steps)
                        #save_image(combined_grid, "/home/renderex/causal_groups/jinyuan.hu/factorizedVAE/factorized_VAE/images/discrete_prior_fitting_res.png")
                    # Visualize generation results
                    c_indices_gen = torch.randint(0, 10, (args.vis_num,), device=device)#换成imagenet时注意把这里的10改成1000（num_classes）
                    index_sample = generate(
                        model.module, c_indices_gen, 256,
                        cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
                        temperature=args.temperature, top_k=args.top_k,
                        top_p=args.top_p, sample_logits=True, 
                        )
                    print(f"index_sample.shape = {index_sample.shape}")
                    index_sample = torch.flatten(index_sample)
                    gen_img = fvae.vq.decode(fvae.vq.quantize.get_codebook_entry(index_sample, shape=(args.vis_num, 16, 16, 8)))
                    gen_grid = make_grid(gen_img, nrow=args.vis_num, normalize=True, value_range=(-1, 1))
                    gen_image = gen_grid.permute(1, 2, 0).cpu().numpy()
                    gen_image = (gen_image * 255).astype(np.uint8)
                    if rank == 0:
                        wandb_logger.log({"Generation Results": wandb.Image(gen_image)}, step=train_steps)
                    save_image(gen_grid, "/home/renderex/causal_groups/jinyuan.hu/factorizedVAE/factorized_VAE/images/discrete_prior_generation_res.png")
                model.train()
                dist.barrier()
                  

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if not args.no_compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "epochs": epoch + 1,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, cloud_checkpoint_path)
                    logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-code-path", type=str, required=True)
    parser.add_argument("--val-code-path", type=str, required=True, help="path to evaluation code, if not specified, use code_path")
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--vis-every", type=int, default=100)
    parser.add_argument("--vis-num", type=int, default=8)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--top-k", type=int, default=0,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
     
    args = parser.parse_args()
    main(args)
