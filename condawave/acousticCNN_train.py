import os
import csv
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===============================
# 1. HDF5数据集
# ===============================

class H5Dataset(Dataset):

    def __init__(self, h5_path):

        self.h5_path = h5_path

        # 这里只读取长度
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['y'])

        # 不提前打开
        self.h5_file = None

    def __len__(self):

        return self.length

    def __getitem__(self, idx):

        # 每个worker第一次访问时再打开
        if self.h5_file is None:

            self.h5_file = h5py.File(
                self.h5_path,
                'r'
            )

            self.X = self.h5_file['X']
            self.y = self.h5_file['y']

        x = self.X[idx]
        y = self.y[idx]-1

        x = torch.tensor(
            x,
            dtype=torch.float32
        ).unsqueeze(0)

        # 标准化
        x = (x - x.mean()) / (x.std() + 1e-6)

        y = torch.tensor(
            y,
            dtype=torch.long
        )

        return x, y


# ===============================
# 2. 模型
# ===============================

class FrequencyBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        c1 = out_ch // 3
        c2 = out_ch // 3
        c3 = out_ch - c1 - c2

        self.conv1 = nn.Conv2d(
            in_ch,
            c1,
            (3,1),
            padding=(1,0)
        )

        self.conv2 = nn.Conv2d(
            in_ch,
            c2,
            (5,1),
            padding=(2,0)
        )

        self.conv3 = nn.Conv2d(
            in_ch,
            c3,
            (7,1),
            padding=(3,0)
        )

        self.bn = nn.BatchNorm2d(out_ch)

        self.act = nn.GELU()

    def forward(self, x):

        x = torch.cat([
            self.conv1(x),
            self.conv2(x),
            self.conv3(x)
        ], dim=1)

        return self.act(self.bn(x))


class TFAttention(nn.Module):

    def __init__(self, ch):

        super().__init__()

        self.freq_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((None,1)),
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid()
        )

        self.time_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,None)),
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return x * self.freq_att(x) * self.time_att(x)


class TemporalBlock(nn.Module):

    def __init__(self, in_ch):

        super().__init__()

        self.lstm = nn.LSTM(
            in_ch,
            64,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):

        B,C,F,T = x.shape

        x = torch.mean(x, dim=2)

        x = x.permute(0,2,1)

        x,_ = self.lstm(x)

        return x.mean(dim=1)


class TFANet(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        self.block1 = FrequencyBlock(32,64)

        self.pool1 = nn.MaxPool2d(2)

        self.block2 = FrequencyBlock(64,128)

        self.pool2 = nn.MaxPool2d(2)

        self.att = TFAttention(128)

        self.temporal = TemporalBlock(128)

        self.fc = nn.Sequential(
            nn.Linear(128,64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64,num_classes)
        )

    def forward(self,x):

        x = self.stem(x)

        x = self.pool1(self.block1(x))

        x = self.pool2(self.block2(x))

        x = self.att(x)

        x = self.temporal(x)

        return self.fc(x)


# ===============================
# 3. train
# ===============================

def train(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0

    preds = []
    labels_all = []

    for x,y in tqdm(loader):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)

        loss = criterion(out,y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds.extend(out.argmax(1).cpu().numpy())

        labels_all.extend(y.cpu().numpy())

    acc = accuracy_score(labels_all, preds)

    return total_loss / len(loader), acc


# ===============================
# 4. evaluate
# ===============================

def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0

    preds = []
    labels_all = []

    with torch.no_grad():

        for x,y in loader:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)

            loss = criterion(out,y)

            total_loss += loss.item()

            preds.extend(out.argmax(1).cpu().numpy())

            labels_all.extend(y.cpu().numpy())

    acc = accuracy_score(labels_all, preds)

    return total_loss / len(loader), acc, preds, labels_all


# ===============================
# 5. main
# ===============================

def main():

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(device)

    # ==========================
    # 数据集
    # ==========================

    train_dataset = H5Dataset(
        "./dataset/train.h5"
    )

    val_dataset = H5Dataset(
        "./dataset/val.h5"
    )

    # ==========================
    # DataLoader
    # ==========================

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # ==========================
    # 模型
    # ==========================

    model = TFANet().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=4e-3
    )

    best_acc = 0

    # ==========================
    # 训练
    # ==========================

    for epoch in range(50):

        print(f"\nEpoch {epoch+1}")

        train_loss, train_acc = train(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_acc, preds, labels = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                "acoustic_best_model.pth"
            )

            print("✅ 保存最佳模型")

    # ==========================
    # 保存日志
    # ==========================

    with open(
        "acoustic_training_log.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc"
        ])

        for i in range(len(history["train_loss"])):

            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i]
            ])

    print("\nBest Acc:", best_acc)

    print("\nConfusion Matrix:")

    print(confusion_matrix(labels, preds))


if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn', force=True)

    main()