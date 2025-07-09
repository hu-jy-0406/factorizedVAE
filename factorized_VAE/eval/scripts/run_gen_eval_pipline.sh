#!/bin/bash

# 设置 CUDA_VISIBLE_DEVICES 环境变量
export CUDA_VISIBLE_DEVICES=6

# 打印当前 CUDA_VISIBLE_DEVICES 设置
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

# 运行 gen_token_for_eval.py
echo "Running gen_token_for_eval.py..."
python /home/renderex/causal_groups/jinyuan.hu/factorizedVAE/factorized_VAE/eval/gen_token_for_eval.py

# 检查 gen_token_for_eval.py 是否成功运行
if [ $? -ne 0 ]; then
    echo "Error: gen_token_for_eval.py failed to execute."
    exit 1
fi

# 运行 eval_discrete_prior_lfq.py
echo "Running eval_discrete_prior_lfq.py..."
python /home/renderex/causal_groups/jinyuan.hu/factorizedVAE/factorized_VAE/eval/eval_discrete_prior_lfq.py

# 检查 eval_discrete_prior_lfq.py 是否成功运行
if [ $? -ne 0 ]; then
    echo "Error: eval_discrete_prior_lfq.py failed to execute."
    exit 1
fi

echo "Both scripts executed successfully."