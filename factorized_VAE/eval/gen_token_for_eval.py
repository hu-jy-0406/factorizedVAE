import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from factorized_VAE.my_models.discrete_prior import DiscretePrior
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12346'
#export PYTHONPATH=/home/renderex/causal_groups/jinyuan.hu/factorizedVAE:$PYTHONPATH

def setup(rank, world_size):
    """
    初始化分布式进程组
    """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """
    销毁分布式进程组
    """
    dist.destroy_process_group()

def generate_tokens_ddp(rank, world_size, model, num_samples, output_dir):
    """
    使用DDP并行生成tokens
    """
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    batch_size = 64
    local_samples = num_samples // world_size  # 每个进程生成的样本数
    generated = []

    with torch.no_grad():
        for _ in tqdm(range(0, local_samples, batch_size), desc=f"Rank {rank} Generating tokens"):
            current_batch_size = min(batch_size, local_samples - len(generated))
            tokens = model.module.generate(num_samples=current_batch_size)  # 调用模型的generate方法
            generated.append(tokens.cpu().numpy())

    # 保存每个进程生成的tokens到文件
    local_output_path = os.path.join(output_dir, f"generated_rank_{rank}.npy")
    np.save(local_output_path, np.concatenate(generated, axis=0))
    print(f"Rank {rank} saved generated tokens to {local_output_path}")

    cleanup()

def merge_generated_tokens(output_dir, world_size, output_file):
    """
    合并所有进程生成的tokens
    """
    all_tokens = []
    for rank in range(world_size):
        local_output_path = os.path.join(output_dir, f"generated_rank_{rank}.npy")
        tokens = np.load(local_output_path)
        all_tokens.append(tokens)
    all_tokens = np.concatenate(all_tokens, axis=0)
    np.save(output_file, all_tokens)
    print(f"All generated tokens saved to {output_file}")

def main(rank, world_size, num_samples, output_dir, output_file):
    """
    主函数，加载模型并启动生成任务
    """
    # --------- Load Model --------- #
    vocab_size = 65536  # set this to match your codebook size
    seq_len = 256
    model = DiscretePrior(vocab_size, seq_len=seq_len, d_model=512, nhead=8, num_layers=8)
    resume_path = "factorized_VAE/DiscretePrior_CIFAR_epoch2100.pth"
    ckpt = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    # --------- Generate Data with DDP ---------- #
    generate_tokens_ddp(rank, world_size, model, num_samples, output_dir)

    # 合并生成的tokens（仅在主进程中执行）
    if rank == 0:
        merge_generated_tokens(output_dir, world_size, output_file)

if __name__ == "__main__":
    num_samples = 60000  # 总共需要生成的样本数
    output_dir = "factorized_VAE/generated_tokens"  # 每个进程生成的tokens保存的目录
    output_file = "factorized_VAE/generated_tokens/all_generated_tokens.npy"  # 最终合并后的文件
    os.makedirs(output_dir, exist_ok=True)

    world_size = torch.cuda.device_count()  # 使用的GPU数量
    mp.spawn(main, args=(world_size, num_samples, output_dir, output_file), nprocs=world_size, join=True)