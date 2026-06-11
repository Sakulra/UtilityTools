# import torch # 如果pytorch安装成功即可导入
# print(torch.cuda.is_available()) # 查看CUDA是否可用
# print(torch.cuda.device_count()) # 查看可用的CUDA数量
# print(torch.version.cuda) # 查看CUDA的版本号

# def count_rows_simple(file_path):
#     """最简单的逐行计数"""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return sum(1 for line in f)

# # 使用
# count = count_rows_simple('./shuffled_dataset/train.csv')
# print(f"文件共有 {count} 行")
import os
import random
import shutil

# 设置源文件夹和目标文件夹路径
src_dir = "F:/shiyan_data/dataset/train/5"   # 替换为你的源文件夹路径
dst_dir = "D:/Document/UtilityTools/condawave/dataset/val/5"     # 替换为目标文件夹路径

# 确保目标文件夹存在
os.makedirs(dst_dir, exist_ok=True)

# 获取所有 .npy 文件列表
files = [f for f in os.listdir(src_dir) if f.endswith('.npy')]

# 随机选择 20% 的文件
num_move = int(len(files) * 0.2)
selected = random.sample(files, num_move)

# 移动文件
for f in selected:
    shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))

print(f"移动完成，共移动 {len(selected)} 个文件到 {dst_dir}")