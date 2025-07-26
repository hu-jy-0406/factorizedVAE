# !/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=/home/renderex/causal_groups/jinyuan.hu/factorizedVAE:$PYTHONPATH
export WANDB_MODE=disabled

torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
--master_port=12355 \
autoregressive/train/train_c2i.py \
--dataset cifar10_code \
--train-code-path /mnt/disk3/jinyuan/CIFAR10-code/fvae/train \
--val-code-path /mnt/disk3/jinyuan/CIFAR10-code/fvae/val \
--image-size 256 \
--num-classes 10 \
--cfg-scale 2.0 \
--global-batch-size 4 \
--epochs 10 \
--log-every 1 \
--vis-every 1 \
--vis-num 4 \
--val-every 1 \
--ckpt-every 1000 \
--no-local-save \
--cloud-save-path /mnt/disk3/jinyuan/ckpts/lamma_gen/ar/pretrain_cifar10
