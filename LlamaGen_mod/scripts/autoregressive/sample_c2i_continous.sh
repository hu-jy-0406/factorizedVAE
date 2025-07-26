# !/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=/home/renderex/causal_groups/jinyuan.hu/factorizedVAE:$PYTHONPATH
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
--master_port=12368 \
autoregressive/sample/sample_c2i_continous_ddp.py \
--gpt-model GPT-B \
--gpt-reg-model GPT-Reg-B \
--gpt-ckpt /mnt/disk3/jinyuan/ckpts/lamma_gen/ar/pretrain_cifar10/2025-07-15-14-52-51/060-GPT-B/checkpoints/0002600.pt \
--gpt-reg-ckpt /mnt/disk3/jinyuan/ckpts/lamma_gen/ar_reg/cifar10/2025-07-26-12-19-26/149-GPT-B/checkpoints/0000100.pt \
--image-size 256 \
--num-classes 10 \
--cfg-scale 2.0 \
--per-proc-batch-size 10 \
--num-fid-samples 10 \
--sample-dir /home/renderex/causal_groups/jinyuan.hu/factorizedVAE/LlamaGen_mod/samples \
--no-compile \

#--gpt-reg-ckpt /mnt/disk3/jinyuan/ckpts/lamma_gen/ar_reg/cifar10/2025-07-21-22-54-18/009-GPT-B/checkpoints/0009000.pt \