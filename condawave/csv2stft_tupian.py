import os
import torch
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==============================
# 参数,转化为图片
# ==============================

CSV_PATH = "acoustic_data.csv"
OUTPUT_DIR = "dataset"

SIGNAL_LENGTH = 4000
CHUNK_SIZE = 5000
TRAIN_RATIO = 0.8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# ==============================
# 创建目录
# ==============================

for split in ["train", "val"]:
    for label in range(1,6):
        os.makedirs(os.path.join(OUTPUT_DIR, split, str(label)), exist_ok=True)

# ==============================
# STFT函数
# ==============================

def signal_to_spectrogram(signal):

    signal = torch.tensor(signal, dtype=torch.float32).to(DEVICE)
    window = torch.hann_window(256).to(DEVICE)

    stft = torch.stft(
        signal,
        n_fft=256,
        hop_length=128,
        win_length=512,
        window= window,
        return_complex=True
    )

    spec = torch.abs(stft)

    spec = torch.log1p(spec)

    spec = spec.cpu().numpy()

    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)

    spec = cv2.resize(spec, (224,224))

    spec = (spec * 255).astype(np.uint8)

    spec = cv2.applyColorMap(spec, cv2.COLORMAP_JET)

    return spec

# ==============================
# 开始处理CSV
# ==============================

print("Start processing CSV...")

reader = pd.read_csv(
    CSV_PATH,
    chunksize=CHUNK_SIZE
)

train_count = 0
val_count = 0

for chunk in reader:

    X = chunk.iloc[:, :SIGNAL_LENGTH].values
    y = chunk.iloc[:, -1].values

    # 分割 train / val
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=TRAIN_RATIO,
        stratify=y,
        random_state=42
    )

    # ==========================
    # 训练集
    # ==========================

    for i in range(len(X_train)):

        signal = X_train[i]

        label = int(y_train[i])

        img = signal_to_spectrogram(signal)

        save_path = os.path.join(
            OUTPUT_DIR,
            "train",
            str(label),
            f"{train_count}.png"
        )

        cv2.imwrite(save_path, img)

        train_count += 1

    # ==========================
    # 验证集
    # ==========================

    for i in range(len(X_val)):

        signal = X_val[i]

        label = int(y_val[i])

        img = signal_to_spectrogram(signal)

        save_path = os.path.join(
            OUTPUT_DIR,
            "val",
            str(label),
            f"{val_count}.png"
        )

        cv2.imwrite(save_path, img)

        val_count += 1

    print(f"Processed: train={train_count}, val={val_count}")

print("Dataset generation finished!")