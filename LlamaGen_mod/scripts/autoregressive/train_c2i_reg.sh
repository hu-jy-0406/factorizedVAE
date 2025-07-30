# !/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/home/guangyi.chen/causal_group/jinyuan.hu/factorizedVAE:$PYTHONPATH
export PYTORCH_SYMBOLIC_SHAPES_DISABLE_WARNINGS=1
export TORCH_DISTRIBUTED_DEBUG=INFO

export WANDB_MODE=disabled

# torchrun \
# --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
# --master_addr=$master_addr --master_port=$master_port \
# autoregressive/train/train_c2i.py "$@"

torchrun \
--nnodes=1 --nproc_per_node=2 --node_rank=0 \
--master_port=12359 \
autoregressive/train/train_c2i_reg.py \
--dataset cifar10_code \
--train-code-path /home/guangyi.chen/causal_group/jinyuan.hu/CIFAR10-latent/fvae/train/ \
--val-code-path /home/guangyi.chen/causal_group/jinyuan.hu/CIFAR10-latent/fvae/val/ \
--image-size 256 \
--num-classes 10 \
--cfg-scale 2.0 \
--global-batch-size 64 \
--vis-num 8 \
--epochs 100 \
--log-every 100 \
--vis-every 1000 \
--val-every 1000 \
--ckpt-every 1000 \
--no-local-save \
--no-save-optimizer \
--cloud-save-path /home/guangyi.chen/causal_group/jinyuan.hu/ckpts/LlamaGen/ar_reg/cifar10 \
--min-ratio 0.3 \
--gpt-ckpt /home/guangyi.chen/causal_group/jinyuan.hu/ckpts/LlamaGen/ar/cifar10/0002600.pt \
#--gpt-ckpt /home/guangyi.chen/causal_group/jinyuan.hu/ckpts/LlamaGen/ar/cifar10-4img/0001000.pt \
# --gpt-ckpt /home/guangyi.chen/causal_group/jinyuan.hu/ckpts/LlamaGen/ar/cifar10/0002600.pt \
#--gpt-reg-ckpt /home/guangyi.chen/causal_group/jinyuan.hu/ckpts/LlamaGen/ar_reg/cifar10/2025-07-29-15-25-45/018-GPT-B/checkpoints/0001000.pt
# --gpt-reg-ckpt /mnt/disk3/jinyuan/ckpts/lamma_gen/test/2025-07-23-17-44-35/046-GPT-Reg-B/checkpoints/0005000.pt

# --gpt-ckpt /mnt/disk3/jinyuan/ckpts/lamma_gen/test/2025-07-23-14-12-26/035-GPT-Reg-B/checkpoints/0001000.pt

# global-batch-size must be greater than 1
# otherwise GroupNorm will fail

#--gpt-ckpt /mnt/disk3/jinyuan/ckpts/lamma_gen/ar_reg/cifar10/2025-07-22-19-31-08/018-GPT-Reg-B/checkpoints/0003200.pt \

# export WANDB_MODE=disabled

# torchrun \
# --nnodes=1 --nproc_per_node=1 --node_rank=0 \
# --master_port=12359 \
# autoregressive/train/train_c2i_reg.py \
# --dataset cifar10_code \
# --train-code-path /mnt/disk3/jinyuan/CIFAR10-latent/fvae/train \
# --val-code-path /mnt/disk3/jinyuan/CIFAR10-latent/fvae/val \
# --image-size 256 \
# --num-classes 10 \
# --cfg-scale 2.0 \
# --global-batch-size 16 \
# --vis-num 16 \
# --epochs 5000 \
# --log-every 100 \
# --vis-every 100 \
# --val-every 100 \
# --ckpt-every 100000000000 \
# --no-local-save \
# --cloud-save-path /mnt/disk3/jinyuan/ckpts/lamma_gen/ar_reg/cifar10/test \
# --min-ratio 0.3 \
# --gpt-ckpt /mnt/disk3/jinyuan/ckpts/lamma_gen/ar/pretrain_cifar10/2025-07-23-10-09-38/025-GPT-B/checkpoints/0001000.pt