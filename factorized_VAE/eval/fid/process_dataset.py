import os
import shutil
from tqdm import tqdm

'''
使用pytorch fid要求数据集图像全部在一个文件夹下，
但实际数据集往往包含多层目录。
本脚本将指定源文件夹中的所有图像文件复制到一个目标文件夹。
'''

# 定义源文件夹和目标文件夹
source_dirs = ["/home/renderex/causal_groups/jinyuan.hu/CIFAR10"]
target_dir = "/home/renderex/causal_groups/jinyuan.hu/CIFAR10-full/"

# 创建目标文件夹
os.makedirs(target_dir, exist_ok=True)

# 遍历源文件夹中的所有文件并复制到目标文件夹
for source_dir in source_dirs:
    for root, _, files in tqdm(os.walk(source_dir)):
        for file in files:
            # 构造源文件路径和目标文件路径
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, file)
            
            # 如果目标文件已存在，添加唯一后缀
            if os.path.exists(target_file):
                base, ext = os.path.splitext(file)  # 分离文件名和扩展名
                counter = 1
                while os.path.exists(target_file):
                    target_file = os.path.join(target_dir, f"{base}_{counter}{ext}")
                    counter += 1
            
            # 复制文件
            shutil.copy2(source_file, target_file)

print(f"All images have been copied to {target_dir}.")