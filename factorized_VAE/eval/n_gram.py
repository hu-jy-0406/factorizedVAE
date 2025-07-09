import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from scipy.stats import entropy
import matplotlib.pyplot as plt
import warnings
import os
import random
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def flatten_tokens(token_array):
    """
    将原始tokens从形状(num, H, W) 转为一维列表
    """
    return token_array.reshape(token_array.shape[0], -1).tolist()

def get_ngram_distribution(sequences, n):
    """
    输入：一个二维token列表（N条序列）
    输出：Counter统计的n-gram频率字典
    """
    counter = Counter()
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:i + n])
            counter[ngram] += 1
    total = sum(counter.values())
    dist = {k: v / total for k, v in counter.items()}
    return dist

def compute_js(p_dist, q_dist):
    """
    计算两个概率分布之间的 JS 散度
    """
    all_keys = set(p_dist.keys()).union(q_dist.keys())
    p = np.array([p_dist.get(k, 1e-8) for k in all_keys])
    q = np.array([q_dist.get(k, 1e-8) for k in all_keys])
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def compute_entropy(dist):
    """
    计算分布的熵
    """
    p = np.array(list(dist.values()))
    return entropy(p)

def plot_top_ngrams(p_dist, q_dist, top_k=20, title=''):
    """
    可视化两个分布中最频繁的 n-gram
    """
    common = sorted(p_dist.items(), key=lambda x: -x[1])[:top_k]
    labels = [' '.join(map(str, k)) for k, _ in common]
    p_vals = [p_dist[k] for k, _ in common]
    q_vals = [q_dist.get(k, 0.0) for k, _ in common]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 4))
    plt.bar(x - width/2, p_vals, width, label='Real')
    plt.bar(x + width/2, q_vals, width, label='Generated')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.title(f'Top-{top_k} {title}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # --------- Load Ground Truth Dataset --------- #
    path = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-VAE-latent/CIFAR10-VAE-discrete-indices.pt"
    real_tokens = torch.load(path)
    B, H, W = real_tokens.shape
    # real_tokens = real_tokens.reshape(B, -1)
    # real_tokens_list = real_tokens.numpy().tolist()
    indices = random.sample(range(B), 1000)
    real_tokens = real_tokens[indices]
    real_tokens = real_tokens.reshape(1000, -1)
    real_tokens_list = real_tokens.numpy().tolist()

    # --------- Load Generated Tokens --------- #
    generated_tokens_path = "factorized_VAE/generated_tokens/all_generated_tokens.npy"
    gen_tokens = np.load(generated_tokens_path)
    gen_tokens_list = gen_tokens.tolist()

    # --------- Compute n-gram Distributions and Metrics --------- #
    for n in [1, 2, 3]:
        print(f"Processing {n}-grams...")
        
        real_dist = get_ngram_distribution(real_tokens_list, n)
        gen_dist = get_ngram_distribution(gen_tokens_list, n)
        js = compute_js(real_dist, gen_dist)
        h_real = compute_entropy(real_dist)
        h_gen = compute_entropy(gen_dist)

        print(f"{n}-gram JS divergence: {js:.4f}")
        print(f"{n}-gram entropy: Real = {h_real:.4f}, Generated = {h_gen:.4f}")

        plot_top_ngrams(real_dist, gen_dist, top_k=15, title=f'{n}-gram Distribution')

if __name__ == "__main__":
    main()
