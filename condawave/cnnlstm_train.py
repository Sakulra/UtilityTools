import os
import json
import time
import pandas as pd  # 需要安装: pip install pandas
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# =========================
# 基础设置（防止卡死）
# =========================
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = True


# =========================
# 模型定义
# =========================
class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, lstm_layers=2, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)              # (B,1,3999) -> 注意：这里原来是3999，需要确认
        x = self.cnn(x)                 # (B,256,L)
        x = x.permute(0, 2, 1)          # (B,L,256)
        x, _ = self.lstm(x)
        x = x[:, -1, :]                 # last timestep
        return self.fc(x)


# =========================
# Dataset
# =========================
class LargeNumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X      # mmap numpy
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# =========================
# 修改：从CSV加载数据
# =========================
def load_training_data_from_csv(csv_path='shuffled_dataset/train.csv'):
    """
    从CSV文件加载训练数据
    CSV格式：前4000列是特征，最后一列是标签（1-5）
    """
    print(f">>> 从CSV加载数据: {csv_path}")
    
    # 使用pandas读取CSV（内存友好，可以分块读取）
    # 如果文件太大，可以使用chunksize参数
    df = pd.read_csv(csv_path)
    
    print(f"数据形状: {df.shape}")
    print(f"列名示例: {df.columns[:5]}...")
    
    # 分离特征和标签
    # 假设最后一列是标签，列名是'label'
    X = df.iloc[:, :-1].values.astype(np.float32)  # 所有行，除最后一列外的所有列
    y = df.iloc[:, -1].values.astype(np.int64)      # 所有行，最后一列
    
    # 标签从1-5转换为0-4（因为CrossEntropyLoss需要从0开始的标签）
    y = y - 1
    
    print(f"特征形状: {X.shape}")
    print(f"标签分布: {np.bincount(y)}")
    print(f"类别数: {len(np.unique(y))}")
    
    # 保存信息
    info = {
        'n_classes': len(np.unique(y)),
        'feature_dim': X.shape[1],
        'n_samples': len(y),
        'class_distribution': np.bincount(y).tolist()
    }
    
    return X, y, info


# =========================
# 如果需要处理超大CSV文件（分块读取）
# =========================
def load_large_csv_in_chunks(csv_path='shuffled_dataset/train.csv', chunk_size=10000):
    """
    对于非常大的CSV文件，分块读取并转换为numpy数组
    """
    print(f">>> 分块加载大CSV: {csv_path}")
    
    # 首先读取第一块以确定列数
    first_chunk = pd.read_csv(csv_path, nrows=1)
    n_features = first_chunk.shape[1] - 1  # 减去标签列
    
    # 计算总行数
    total_rows = sum(1 for _ in open(csv_path)) - 1  # 减去表头
    
    print(f"总行数: {total_rows}, 特征数: {n_features}")
    
    # 预分配数组
    X = np.zeros((total_rows, n_features), dtype=np.float32)
    y = np.zeros(total_rows, dtype=np.int64)
    
    # 分块读取
    start_idx = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk_size_actual = len(chunk)
        X[start_idx:start_idx+chunk_size_actual] = chunk.iloc[:, :-1].values
        y[start_idx:start_idx+chunk_size_actual] = chunk.iloc[:, -1].values
        start_idx += chunk_size_actual
        print(f"已加载 {start_idx}/{total_rows} 行")
    
    # 标签转换为0-based
    y = y - 1
    
    info = {
        'n_classes': len(np.unique(y)),
        'feature_dim': n_features,
        'n_samples': total_rows
    }
    
    return X, y, info


# =========================
# DataLoader
# =========================
def create_dataloaders(X, y, batch_size=64, val_ratio=0.1):
    n = len(y)
    idx = np.random.permutation(n)
    split = int(val_ratio * n)

    val_idx = idx[:split]
    train_idx = idx[split:]

    train_ds = LargeNumpyDataset(X[train_idx], y[train_idx])
    val_ds   = LargeNumpyDataset(X[val_idx], y[val_idx])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,          # Windows下建议用0
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader


# =========================
# 训练 & 验证
# =========================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total, correct, loss_sum = 0, 0, 0

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()
        pred = out.argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()

        if i == 0:
            print(f"[Epoch {epoch}] First batch OK, loss={loss.item():.4f}")

        if i % 100 == 0:
            print(f"Epoch {epoch} | Batch {i}/{len(loader)}")

    return loss_sum / len(loader), 100. * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0
    preds, targets = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        loss = criterion(out, y)

        loss_sum += loss.item()
        pred = out.argmax(1)

        total += y.size(0)
        correct += (pred == y).sum().item()

        preds.extend(pred.cpu().numpy())
        targets.extend(y.cpu().numpy())

    return loss_sum / len(loader), 100. * correct / total, preds, targets


# =========================
# 主函数
# =========================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # 从CSV加载数据
    # 如果文件不大，使用这个
    # X, y, info = load_training_data_from_csv('shuffled_dataset/train.csv')
    
    # 如果文件很大，使用这个（取消下面的注释）
    X, y, info = load_large_csv_in_chunks('./shuffled_dataset/train.csv', chunk_size=10000)

    train_loader, val_loader = create_dataloaders(
        X, y,
        batch_size=64,      # 可以根据GPU内存调整
        val_ratio=0.1       # 从训练集中再分出10%作为验证集
    )

    model = CNNLSTMClassifier(
        input_dim=4000,      # 修改为4000
        num_classes=info['n_classes']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0

    for epoch in range(1, 51):
        print(f"\n===== Epoch {epoch} =====")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )

        val_loss, val_acc, preds, targets = validate(
            model, val_loader, criterion, device
        )

        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("✓ Saved best model")

    print("\nFinal Report:")
    print(classification_report(targets, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(targets, preds))


if __name__ == "__main__":
    main()