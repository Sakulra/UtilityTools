import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# 信号参数
# -------------------------

FS = 50000
NFFT = 512
HOP = 128

window = torch.hann_window(NFFT).to(device)

# 频率范围
FREQ_MIN = 11000
FREQ_MAX = 21000

# 数据划分比例
VAL_RATIO = 0.2


# -------------------------
# STFT -> feature
# -------------------------

def stft_to_feature(signal):

    signal = torch.tensor(signal, dtype=torch.float32).to(device)

    stft = torch.stft(
        signal,
        n_fft=NFFT,
        hop_length=HOP,
        win_length=NFFT,
        window=window,
        return_complex=True
    )

    spec = torch.abs(stft)

    # 频率轴
    freqs = torch.linspace(0, FS/2, spec.shape[0]).to(device)

    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)

    spec = spec[mask]

    spec = torch.log1p(spec)

    spec = spec.cpu().numpy()

    spec = cv2.resize(spec, (224,224))

    spec = (spec - spec.min())/(spec.max()-spec.min()+1e-6)

    return spec.astype(np.float32)


# -------------------------
# CSV processing
# -------------------------

def process_csv(csv_path, output_dir, chunk_size=20000):

    reader = pd.read_csv(csv_path, chunksize=chunk_size)

    sample_id = 0

    for chunk in reader:

        signals = chunk.iloc[:,:4000].values
        labels = chunk.iloc[:,4000].values

        for i in tqdm(range(len(signals))):

            signal = signals[i]
            label = int(labels[i])

            feature = stft_to_feature(signal)

            # 随机划分 train / val
            if np.random.rand() < VAL_RATIO:
                split = "val"
            else:
                split = "train"

            save_dir = f"{output_dir}/{split}/{label}"

            os.makedirs(save_dir, exist_ok=True)

            np.save(f"{save_dir}/{sample_id}.npy", feature)

            sample_id += 1


# -------------------------
# main
# -------------------------

if __name__ == "__main__":

    process_csv(
        csv_path="add_labeled_4000.csv",
        output_dir="dataset"
    )