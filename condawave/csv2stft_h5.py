import os
import numpy as np
import pandas as pd
import torch
import h5py
from tqdm import tqdm

# ======================
# 参数
# ======================

FS = 200000
NFFT = 256
HOP = 64

FREQ_MIN = 11000
FREQ_MAX = 21000

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ======================
# 全局 window
# ======================

window = torch.hann_window(NFFT)

# ======================
# STFT
# ======================

def stft_to_feature(signal):

    signal = torch.tensor(signal, dtype=torch.float32)

    stft = torch.stft(
        signal,
        n_fft=NFFT,
        hop_length=HOP,
        win_length=NFFT,
        window=window,
        return_complex=True
    )

    spec = torch.abs(stft)

    # 频率裁剪
    freqs = torch.linspace(0, FS/2, spec.shape[0])

    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)

    spec = spec[mask]

    # log
    spec = torch.log1p(spec)

    spec = spec.numpy()

    # normalize
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)

    return spec.astype(np.float16)

# ======================
# 创建可扩展HDF5
# ======================

def create_h5(path, feature_shape):

    h5 = h5py.File(path, 'w')

    X = h5.create_dataset(
        'X',
        shape=(0, *feature_shape),
        maxshape=(None, *feature_shape),
        dtype=np.float16,
        chunks=True
    )

    y = h5.create_dataset(
        'y',
        shape=(0,),
        maxshape=(None,),
        dtype=np.int64,
        chunks=True
    )

    return h5, X, y

# ======================
# append数据
# ======================

def append_h5(X_ds, y_ds, feature, label):

    n = X_ds.shape[0]

    X_ds.resize(n + 1, axis=0)
    y_ds.resize(n + 1, axis=0)

    X_ds[n] = feature
    y_ds[n] = label

# ======================
# 主处理
# ======================

def process_csv(csv_path, output_dir, chunk_size=5000):

    os.makedirs(output_dir, exist_ok=True)

    reader = pd.read_csv(csv_path, chunksize=chunk_size)

    initialized = False

    for chunk_id, chunk in enumerate(reader):

        print(f"\nProcessing chunk {chunk_id}")

        signals = chunk.iloc[:, :4000].values
        labels = chunk.iloc[:, 4000].values.astype(np.int64)

        for i in tqdm(range(len(signals))):

            feature = stft_to_feature(signals[i])
            label = labels[i]

            # 第一次初始化H5
            if not initialized:

                feature_shape = feature.shape

                train_h5, train_X, train_y = create_h5(
                    os.path.join(output_dir, 'train.h5'),
                    feature_shape
                )

                val_h5, val_X, val_y = create_h5(
                    os.path.join(output_dir, 'val.h5'),
                    feature_shape
                )

                test_h5, test_X, test_y = create_h5(
                    os.path.join(output_dir, 'test.h5'),
                    feature_shape
                )

                initialized = True

            # 数据划分
            r = np.random.rand()

            if r < TRAIN_RATIO:

                append_h5(train_X, train_y, feature, label)

            elif r < TRAIN_RATIO + VAL_RATIO:

                append_h5(val_X, val_y, feature, label)

            else:

                append_h5(test_X, test_y, feature, label)

    train_h5.close()
    val_h5.close()
    test_h5.close()

    print("Done.")

# ======================
# main
# ======================

if __name__ == "__main__":

    process_csv(
        csv_path="add_labeled_4000.csv",
        output_dir="F:/shiyan_data/dataset",
        chunk_size=5000
    )