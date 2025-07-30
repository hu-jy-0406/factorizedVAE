# !/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/guangyi.chen/causal_group/jinyuan.hu/factorizedVAE:$PYTHONPATH
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
--master_port=12368 \
autoregressive/sample/sample_c2i_continous_ddp_debug.py \
--gpt-model GPT-B \
--gpt-reg-model GPT-Reg-B \
--gpt-ckpt /home/guangyi.chen/causal_group/jinyuan.hu/ckpts/LlamaGen/ar/cifar10-4img/0001000.pt \
--gpt-reg-ckpt /home/guangyi.chen/causal_group/jinyuan.hu/ckpts/LlamaGen/ar_reg/cifar10-4img/2025-07-30-10-58-25/041-GPT-B/checkpoints/0001000.pt \
--image-size 256 \
--num-classes 10 \
--cfg-scale 2.0 \
--per-proc-batch-size 4 \
--num-fid-samples 4 \
--sample-dir /home/guangyi.chen/causal_group/jinyuan.hu/factorizedVAE/LlamaGen_mod/samples \
--no-compile \
--info 4img \
--dataset cifar10_code \
--train-code-path /home/guangyi.chen/causal_group/jinyuan.hu/CIFAR10-latent/fvae/train \
--val-code-path /home/guangyi.chen/causal_group/jinyuan.hu/CIFAR10-latent/fvae/val \

#--gpt-reg-ckpt /home/guangyi.chen/causal_group/jinyuan.hu/ckpts/LlamaGen/ar_reg/cifar10/2025-07-29-14-33-57/017-GPT-B/checkpoints/0001000.pt \

#--gpt-reg-ckpt /mnt/disk3/jinyuan/ckpts/lamma_gen/ar_reg/cifar10/2025-07-21-22-54-18/009-GPT-B/checkpoints/0009000.pt \