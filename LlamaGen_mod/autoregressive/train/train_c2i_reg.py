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

from LlamaGen.utils.logger import create_logger
from LlamaGen.utils.distributed import init_distributed_mode
from LlamaGen.utils.ema import update_ema, requires_grad
from LlamaGen.dataset.build import build_dataset
from LlamaGen.autoregressive.models.gpt import GPT_models
from LlamaGen.autoregressive.models.gpt_reg import GPT_Reg_models
from LlamaGen.autoregressive.models.generate import generate, generate_continous_code

import torch._dynamo
torch._dynamo.config.suppress_errors = True #important! This prevents torch._dynamo from raising errors when it encounters unsupported operations, which can happen with certain model architectures or configurations.

import logging
logging.getLogger("torch").setLevel(logging.ERROR)


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
        wandb_logger = wandb.init(project="LlamaGen_Reg",
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

    # fvae = FVAE()
    # ckpt_path = "/mnt/disk3/jinyuan/ckpts/fvae/full/fvae_full_3loss+epoch10.pth"
    # ckpt = torch.load(ckpt_path)
    # fvae.load_state_dict(ckpt["model"])    
    
    # Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    gpt_reg_model = GPT_Reg_models[args.gpt_reg_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        no_mem=args.no_mem
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in gpt_reg_model.parameters()):,}")

    if args.ema:
        ema = deepcopy(gpt_reg_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    # Setup optimizer
    optimizer = creat_optimizer(gpt_reg_model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

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
    
    # create and load gpt model for inference
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
    
    
    # Prepare models for training:
    if args.gpt_reg_ckpt:
        checkpoint = torch.load(args.gpt_reg_ckpt, map_location="cpu")
        gpt_reg_model.load_state_dict(checkpoint["model"], strict=False)
        if args.ema:
            ema.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"])
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("Optimizer state loaded successfully.")
            except Exception as e:
                print(f"Failed to load optimizer state: {e}")
        if "steps" in checkpoint:
            train_steps = checkpoint["steps"]
        train_steps = checkpoint["steps"] if "steps" in checkpoint else 0
        start_epoch = checkpoint["epochs"] if "epochs" in checkpoint else 0
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
    else:
        train_steps = 0
        start_epoch = 0
        
    logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    
    if args.ema:
        update_ema(ema, gpt_reg_model, decay=0)  # Ensure EMA is initialized with synced weights

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        gpt_reg_model = torch.compile(gpt_reg_model) # requires PyTorch 2.0        
    
    gpt_reg_model = DDP(gpt_reg_model.to(device), device_ids=[args.gpu])
    gpt_reg_model.train()  # important! This enables embedding dropout for classifier-free guidance
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
        for x, y in vis_loader:
            x = x.to(device, non_blocking=True) # (64, 256, 16)
            y = y.to(device, non_blocking=True) # (64)
            
            if args.no_mem==True:
                print("no_mem is True, mem is None")
                mem_gt = None
            else:
                mem_gt = x.unsqueeze(2).permute(0,3,1,2) # from (B, 256, z_dim=16) to (B, z_dim=16, 256, 1)
                mem_gt = gpt_reg_model.module.fvae.get_quant_from_vae_latent(mem_gt)  # (B, z_dim=16, 256, 1) -> (B, z_dim=8, 256, 1)
                mem_gt = mem_gt.permute(0,2,3,1) # (B, 256, 1, z_dim=8)
                mem_gt = mem_gt.squeeze(2).detach()  # (B, 256, z_dim=8)
            
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                _, mse_loss, ploss, _, _ = gpt_reg_model(codes=x, cond_idx=y, mem=mem_gt, targets=x)
            loss = mse_loss + ploss
            if rank == 0:
                wandb_logger.log({"step_train_loss": loss.item()}, step=train_steps)         
            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(gpt_reg_model.parameters(), args.max_grad_norm)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            if args.ema:
                update_ema(ema, gpt_reg_model.module._orig_mod if not args.no_compile else gpt_reg_model.module)

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
            # if train_steps % args.val_every == 0 and train_steps > 0:
            #     logger.info(f"Begining evaluation at step{train_steps}, epoch{epoch}")
            #     model.eval()#这里不能加model.eval()，加之后model.forward()会报错
            #     val_steps = 0
            #     total_val_loss = 0.0
            #     with torch.no_grad():
            #         for x, y in val_loader:
            #             x = x.to(device, non_blocking=True)#(256, 1, 1, 256)
            #             y = y.to(device, non_blocking=True)#(256, 1)
                        
            #             if args.no_mem==True:
            #                 print(f"no_mem is True, mem is set to None")
            #                 mem = None
            #             else:
            #                 mem = x.unsqueeze(2).permute(0,3,1,2) # from (B, 256, z_dim=16) to (B, z_dim=16, 256, 1)
            #                 mem = model.module.fvae.get_quant_from_vae_latent(mem)  # (B, z_dim=16, 256, 1) -> (B, z_dim=8, 256, 1)
            #                 mem = mem.permute(0,2,3,1) # (B, 256, 1, z_dim=8)
            #                 mem = mem.squeeze(2).detach()  # (B, 256, z_dim=8)
            
            #             with torch.cuda.amp.autocast(dtype=ptdtype):  
            #                 _, mse_loss, ploss, _, _ = model(codes=x, cond_idx=y, mem=mem, input_pos=torch.arange(256), targets=x)
            #             loss = mse_loss + ploss
            #             total_val_loss += loss.item()
            #             val_steps += 1
            #         print(f"total_val_loss = {total_val_loss}, val_steps = {val_steps}")
            #     avg_val_loss = torch.tensor(total_val_loss / val_steps, device=device)
                
            #     if rank == 0:
            #         wandb_logger.log({"step_val_loss": avg_val_loss.item()}, step=train_steps)
                    
            #     dist.all_reduce(avg_val_loss, op=dist.ReduceOp.SUM)
            #     avg_val_loss = avg_val_loss.item() / dist.get_world_size()
            #     logger.info(f"(step={train_steps:07d}) Val Loss: {avg_val_loss:.4f}")
            #     if rank == 0:
            #         wandb_logger.log({"total_val_loss": total_val_loss}, step=train_steps)
            #         wandb_logger.log({"val_loss": avg_val_loss}, step=train_steps)
            #     model.train()
                
            # Visualize fitting result and generation result
            '''
            判断模型对训练样本的拟合情况以及生成新样本的能力
            '''
            if train_steps % args.vis_every == 0 and train_steps > 0:
                logger.info(f"Begining visualization at step{train_steps}, epoch{epoch}")
                gpt_reg_model.eval()
                with torch.no_grad():
                    # Visualize fitting results
                    for x, y in vis_loader:
                        x = x.to(device, non_blocking=True)#(b, 1, 1, 256)
                        y = y.to(device, non_blocking=True)#(b, 1)
                        
                        if args.no_mem==True:
                            mem_gt = None
                        else:
                            mem_gt = x.unsqueeze(2).permute(0,3,1,2) # from (B, 256, z_dim=16) to (B, z_dim=16, 256, 1)
                            mem_gt = gpt_reg_model.module.fvae.get_quant_from_vae_latent(mem_gt)  # (B, z_dim=16, 256, 1) -> (B, z_dim=8, 256, 1)
                            mem_gt = mem_gt.permute(0,2,3,1) # (B, 256, 1, z_dim=8)
                            mem_gt = mem_gt.squeeze(2).detach()  # (B, 256, z_dim=8)
            
                        # z_indices = x.reshape(x.shape[0], -1)# (b, 256)
                        # c_indices = y.reshape(-1)# ([256])
                        # assert z_indices.shape[0] == c_indices.shape[0]
                        with torch.cuda.amp.autocast(dtype=ptdtype):  
                            _, mse_loss, ploss, gt_img, recon_img = gpt_reg_model(codes=x, cond_idx=y, mem=mem_gt, input_pos=torch.arange(256), targets=x)
                            
                        recon_grid = make_grid(recon_img, nrow=args.vis_num, normalize=True, value_range=(-1, 1))
                        gt_grid = make_grid(gt_img, nrow=args.vis_num, normalize=True, value_range=(-1, 1))
                        combined_grid = torch.cat((gt_grid, recon_grid), dim=1)
                        combined_image = combined_grid.permute(1, 2, 0).cpu().to(torch.float32).numpy()
                        combined_image = (combined_image * 255).astype(np.uint8)
                        if rank == 0:
                            wandb_logger.log({"Reconstruction Comparison": wandb.Image(combined_image)}, step=train_steps)
                        save_image(combined_grid, "/home/renderex/causal_groups/jinyuan.hu/factorizedVAE/factorized_VAE/images/GPT-Reg-fit-res.png")
                        
                        # ++++++++++++++++++ Visualize generation results +++++++++++++++++++++ #
                        # ------------------ Generate with ground truth memory ------------------- #
                        c_indices_gen = torch.randint(0, 10, (args.vis_num,), device=device)#换成imagenet时注意把这里的10改成1000（num_classes）
                        
                        code_sample = generate_continous_code(
                            model=gpt_reg_model.module, cond=c_indices_gen, mem=mem_gt, max_new_codes=latent_size ** 2,
                            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval)
                        
                        b, l, c = code_sample.shape
                        h = w = int(np.sqrt(l))
                                                
                        code_sample = code_sample.reshape(b, h, w, c).permute(0, 3, 1, 2)  # (b, 16, 16, 16)
                        img_sample = gpt_reg_model.module.fvae.kl.decode(code_sample)  # (b, 3, 256, 256)
                        gen_grid1 = make_grid(img_sample, nrow=args.vis_num, normalize=True, value_range=(-1, 1))
                        mem_gt_grid = gt_grid
                        # combined_grid2 = torch.cat((mem_gt_grid, gen_grid1), dim=1)
                        # combined_image2 = combined_grid2.permute(1, 2, 0).cpu().numpy()
                        # combined_image2 = (combined_image2 * 255).astype(np.uint8)
                        
                        # if rank == 0:
                        #     wandb_logger.log({"Generation Results": wandb.Image(combined_image2)}, step=train_steps)
                        # save_image(gen_grid1, "/home/renderex/causal_groups/jinyuan.hu/factorizedVAE/factorized_VAE/images/GPT-Reg-gen-res.png")
                        # -------------------------------------------------------------------------- #
                        
                        # ------------------- Generate with sampled memory -------------------- #
                        c_indices = torch.randint(0, args.num_classes, (b,), device=device)
                        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
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
                        vq_code = gpt_reg_model.module.fvae.vq.quantize.get_codebook_entry(index_sample, shape=(b, 16, 16, 8)) # (b, 8, 16, 16)
                        vq_code_dec = gpt_reg_model.module.fvae.vq.decode(vq_code)
                        mem_gen_grid = make_grid(vq_code_dec, nrow=args.vis_num, normalize=True, value_range=(-1, 1))
                        mem_gen = vq_code.permute(0, 2, 3, 1).reshape(b, latent_size ** 2, -1)  # (b, 256, 8)
                        
                        code_sample2 = generate_continous_code(
                            model=gpt_reg_model.module, cond=c_indices_gen, mem=mem_gen, max_new_codes=latent_size ** 2,
                            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval)
                        
                        b, l, c = code_sample2.shape
                        h = w = int(np.sqrt(l))
                                                
                        code_sample2 = code_sample2.reshape(b, h, w, c).permute(0, 3, 1, 2)  # (b, 16, 16, 16)
                        img_sample2 = gpt_reg_model.module.fvae.kl.decode(code_sample2)  # (b, 3, 256, 256)
                        gen_grid2 = make_grid(img_sample2, nrow=args.vis_num, normalize=True, value_range=(-1, 1))
                        
                        mem_mean = (mem_gen + mem_gt) / 2 # (B, 256, 8)
                        mean_mean_code = mem_mean.reshape(b, h, w, -1).permute(0, 3, 1, 2)  # (b, 8, 16, 16)
                        mem_mean_dec = gpt_reg_model.module.fvae.vq.decode(mean_mean_code)
                        mem_mean_grid = make_grid(mem_mean_dec, nrow=args.vis_num, normalize=True, value_range=(-1, 1))
                        
                        
                        code_sample3 = generate_continous_code(
                            model=gpt_reg_model.module, cond=c_indices_gen, mem=mem_mean, max_new_codes=latent_size ** 2,
                            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval)
                        b, l, c = code_sample3.shape
                        h = w = int(np.sqrt(l))
                        code_sample3 = code_sample3.reshape(b, h, w, c).permute(0, 3, 1, 2)  # (b, 16, 16, 16)
                        img_sample3 = gpt_reg_model.module.fvae.kl.decode(code_sample3)  # (b, 3, 256, 256)
                        gen_grid3 = make_grid(img_sample3, nrow=args.vis_num, normalize=True, value_range=(-1, 1))
                        
                        combined_grid3 = torch.cat((mem_gt_grid, gen_grid1, mem_gen_grid, gen_grid2, mem_mean_grid, gen_grid3), dim=1)
                        combined_image3 = combined_grid3.permute(1, 2, 0).cpu().numpy()
                        combined_image3 = (combined_image3 * 255).astype(np.uint8)
                        print("l2 gap_between mem_gt and mem_gen:", torch.nn.functional.mse_loss(mem_gt, mem_gen).item())
                        
                        if rank == 0:
                            wandb_logger.log({"Generation with Sampled Memory": wandb.Image(combined_image3)}, step=train_steps)
                        save_image(combined_grid3, "/home/renderex/causal_groups/jinyuan.hu/factorizedVAE/factorized_VAE/images/GPT-Reg-gen-res.png")
                        
                        # -------------------------------------------------------------------------- #
                        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
                        
                        
                        
                gpt_reg_model.train()
                dist.barrier()
                  

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if not args.no_compile:
                        model_weight = gpt_reg_model.module._orig_mod.state_dict()
                    else:
                        model_weight = gpt_reg_model.module.state_dict()  
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

    gpt_reg_model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-code-path", type=str, required=True)
    parser.add_argument("--val-code-path", type=str, required=True, help="path to evaluation code, if not specified, use code_path")
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', default='False', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-reg-model", type=str, choices=list(GPT_Reg_models.keys()), default="GPT-Reg-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-reg-ckpt", type=str, default=None)
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
    parser.add_argument("--no-mem", action='store_true', default='False', help="whether to use memory projection in the model")
    
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--top-k", type=int, default=0,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--from-fsdp", action='store_true')
     
    args = parser.parse_args()
    main(args)
