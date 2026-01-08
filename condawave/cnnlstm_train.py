import os
import json
import time
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
        x = x.unsqueeze(1)              # (B,1,3999)
        x = self.cnn(x)                 # (B,256,L)
        x = x.permute(0, 2, 1)          # (B,L,256)
        x, _ = self.lstm(x)
        x = x[:, -1, :]                 # last timestep
        return self.fc(x)


# =========================
# Dataset（关键）
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
# 数据加载（mmap）
# =========================
def load_training_data(data_dir='data'):
    print(">>> 使用 mmap 加载数据")
    data = np.load(os.path.join(data_dir, 'train_data.npz'), mmap_mode='r')
    X = data['X']
    y = data['y']

    with open(os.path.join(data_dir, 'data_info.json'), 'r') as f:
        info = json.load(f)

    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"Classes: {info['n_classes']}")
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

    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=2,
    #     pin_memory=True,
    #     persistent_workers=True,
    #     prefetch_factor=2
    # )
    train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=0,          # ✅ 关键
    pin_memory=True
    )

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=2,
    #     pin_memory=True
    # )
    val_loader = DataLoader(
    val_ds,
    batch_size=64,
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

    X, y, info = load_training_data()

    train_loader, val_loader = create_dataloaders(
        X, y,
        batch_size=64,      # ★关键
        val_ratio=0.1
    )

    model = CNNLSTMClassifier(
        input_dim=3999,
        num_classes=info['n_classes']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0

    for epoch in range(1, 11):
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