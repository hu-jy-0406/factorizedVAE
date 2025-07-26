# !/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=/home/renderex/causal_groups/jinyuan.hu/factorizedVAE:$PYTHONPATH

# torchrun \
# --nnodes=1 --nproc_per_node=8 --node_rank=0 \
# --master_port=12345 \
# autoregressive/sample/sample_c2i_ddp.py \
# --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
# "$@"

torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
--master_port=12368 \
autoregressive/sample/sample_c2i_ddp.py \
--gpt-model GPT-B \
--gpt-ckpt /mnt/disk3/jinyuan/ckpts/lamma_gen/ar/pretrain_cifar10/2025-07-23-10-09-38/025-GPT-B/checkpoints/0001000.pt \
--image-size 256 \
--num-classes 10 \
--cfg-scale 2.0 \
--per-proc-batch-size 32 \
--num-fid-samples 10 \
--sample-dir /home/renderex/causal_groups/jinyuan.hu/factorizedVAE/LlamaGen_mod/samples \
