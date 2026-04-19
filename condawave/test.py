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
