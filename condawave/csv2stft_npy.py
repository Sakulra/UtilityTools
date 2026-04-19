import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# 信号参数（已根据 12k-20kHz 信号优化）
# -------------------------
FS = 50000
NFFT = 256  # 减小窗口，提高时间分辨率，减少 resize 时的拉伸感
HOP = 64    # 减小跳步，增加时间帧数，使图像更细腻
# 注意：window 需要在函数内或根据新的 NFFT 重新生成

# 频率范围
FREQ_MIN = 11000
FREQ_MAX = 21000

# 数据划分比例
VAL_RATIO = 0.2

# -------------------------
# STFT -> feature (更正版)
# -------------------------
def stft_to_feature(signal):
    # 1. 转换为 tensor 并移至设备
    signal = torch.tensor(signal, dtype=torch.float32).to(device)

    # 2. 执行 STFT
    # 使用与 NFFT 匹配的 window
    curr_window = torch.hann_window(NFFT).to(device)
    
    stft = torch.stft(
        signal,
        n_fft=NFFT,
        hop_length=HOP,
        win_length=NFFT,
        window=curr_window,
        return_complex=True
    )

    # 3. 取幅值
    spec = torch.abs(stft)

    # 4. 频率裁剪：只保留 11kHz 到 21kHz
    # 频率轴分辨率: FS / NFFT = 50000 / 256 ≈ 195.3 Hz
    freqs = torch.linspace(0, FS/2, spec.shape[0]).to(device)
    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    spec = spec[mask]

    # 5. 对数压缩：压缩动态范围，强化弱信号特征
    spec = torch.log1p(spec)

    # 6. 转回 CPU 进行图像处理
    spec_np = spec.cpu().numpy()

    # 7. 归一化：将特征缩放到 [0, 1]
    s_min = spec_np.min()
    s_max = spec_np.max()
    spec_np = (spec_np - s_min) / (s_max - s_min + 1e-6)

    # 8. 调整尺寸：缩放到模型输入要求的 224x224
    # 使用 INTER_CUBIC 插值在放大时获得更平滑的边缘
    # spec_resized = cv2.resize(spec_np, (224, 224), interpolation=cv2.INTER_CUBIC)
    # return spec_resized.astype(np.float32)

    #不强制resize，就返回原图
    return spec_np.astype(np.float32)


# -------------------------
# CSV processing
# -------------------------
def process_csv(csv_path, output_dir, chunk_size=20000):
    # 检查输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reader = pd.read_csv(csv_path, chunksize=chunk_size)
    sample_id = 0

    for chunk in reader:
        # 假设前 4000 列是信号点，第 4001 列是标签
        signals = chunk.iloc[:, :4000].values
        labels = chunk.iloc[:, 4000].values

        for i in tqdm(range(len(signals)), desc=f"Processing chunk (ID starts at {sample_id})"):
            signal = signals[i]
            label = int(labels[i])

            # 特征提取
            feature = stft_to_feature(signal)

            # 随机划分 train / val
            split = "val" if np.random.rand() < VAL_RATIO else "train"
            
            # 存储路径
            save_path = os.path.join(output_dir, split, str(label))
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            # 保存为 npy 文件
            np.save(os.path.join(save_path, f"{sample_id}.npy"), feature)
            sample_id += 1


# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    # 请确保文件名与路径正确
    process_csv(
        csv_path="add_labeled_4000.csv",
        output_dir="F:/shiyan_data/dataset"
    )