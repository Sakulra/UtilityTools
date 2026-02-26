import torch
print(torch.cuda.is_available())  # 如果返回 False，表示没有可用 GPU
print(torch.cuda.device_count())  # GPU 数量
print(torch.version.cuda)         # CUDA 版本