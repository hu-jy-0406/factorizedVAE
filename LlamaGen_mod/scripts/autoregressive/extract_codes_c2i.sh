# !/bin/bash
set -x

# torchrun \
# --nnodes=1 --nproc_per_node=8 --node_rank=0 \
# --master_port=12335 \
# autoregressive/train/extract_codes_c2i.py "$@"

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=/home/renderex/causal_groups/jinyuan.hu/factorizedVAE:$PYTHONPATH

torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
--master_port=12335 \
autoregressive/train/extract_codes_c2i.py \
--vq-ckpt /mnt/disk3/jinyuan/ckpts/lamma_gen/vq_vae/vq_ds16_c2i.pt \
--data-path /mnt/disk3/jinyuan/CIFAR10/train/ \
--code-path /mnt/disk3/jinyuan/CIFAR10-latent/fvae/train/ \
--dataset cifar10 \
--dataset-type train \
--image-size 256 \
--modeltype fvae

# torchrun \
# --nnodes=1 --nproc_per_node=1 --node_rank=0 \
# --master_port=12335 \
# autoregressive/train/extract_codes_c2i_backup.py \
# --vq-ckpt /home/renderex/causal_groups/jinyuan.hu/ckpts/lamma_gen/vq_vae/vq_ds16_c2i.pt \
# --data-path /home/renderex/causal_groups/jinyuan.hu/CIFAR10/ \
# --code-path /home/renderex/causal_groups/jinyuan.hu/CIFAR10-code/lfq_vae/ \
# --dataset cifar10 \
# --image-size 256