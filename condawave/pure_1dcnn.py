import os
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# =========================
# 基础设置
# =========================
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = True


# =========================
# Pure 1D-CNN Model
# =========================
class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv1d(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 2
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 3
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Dropout(dropout)
        )

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        # 输入:
        # x -> (B, 4000)

        x = x.unsqueeze(1)

        # -> (B,1,4000)

        x = self.features(x)

        # -> (B,256,L)

        x = self.gap(x)

        # -> (B,256,1)

        x = x.squeeze(-1)

        # -> (B,256)

        x = self.classifier(x)

        return x


# =========================
# Dataset
# =========================
class LargeNumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# =========================
# 从CSV加载数据
# =========================
def load_training_data_from_csv(csv_path='shuffled_dataset/train.csv'):

    print(f">>> 从CSV加载数据: {csv_path}")

    df = pd.read_csv(csv_path)

    print(f"数据形状: {df.shape}")

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    # 标签转为0开始
    y = y - 1

    print(f"特征形状: {X.shape}")
    print(f"标签分布: {np.bincount(y)}")
    print(f"类别数: {len(np.unique(y))}")

    info = {
        'n_classes': len(np.unique(y)),
        'feature_dim': X.shape[1],
        'n_samples': len(y),
        'class_distribution': np.bincount(y).tolist()
    }

    return X, y, info


# =========================
# 大CSV分块读取
# =========================
def load_large_csv_in_chunks(
        csv_path='shuffled_dataset/train.csv',
        chunk_size=10000):

    print(f">>> 分块加载大CSV: {csv_path}")

    first_chunk = pd.read_csv(csv_path, nrows=1)

    n_features = first_chunk.shape[1] - 1

    total_rows = sum(1 for _ in open(csv_path)) - 1

    print(f"总行数: {total_rows}")
    print(f"特征数: {n_features}")

    X = np.zeros((total_rows, n_features), dtype=np.float32)
    y = np.zeros(total_rows, dtype=np.int64)

    start_idx = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):

        chunk_size_actual = len(chunk)

        X[start_idx:start_idx + chunk_size_actual] = \
            chunk.iloc[:, :-1].values

        y[start_idx:start_idx + chunk_size_actual] = \
            chunk.iloc[:, -1].values

        start_idx += chunk_size_actual

        print(f"已加载 {start_idx}/{total_rows}")

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
def create_dataloaders(
        X,
        y,
        batch_size=64,
        val_ratio=0.1):

    n = len(y)

    idx = np.random.permutation(n)

    split = int(val_ratio * n)

    val_idx = idx[:split]
    train_idx = idx[split:]

    train_ds = LargeNumpyDataset(
        X[train_idx],
        y[train_idx]
    )

    val_ds = LargeNumpyDataset(
        X[val_idx],
        y[val_idx]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
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
# 训练
# =========================
def train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        scaler,
        device,
        epoch):

    model.train()

    total = 0
    correct = 0
    loss_sum = 0

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
            print(
                f"[Epoch {epoch}] "
                f"First batch OK, "
                f"loss={loss.item():.4f}"
            )

        if i % 100 == 0:
            print(f"Epoch {epoch} | Batch {i}/{len(loader)}")

    train_loss = loss_sum / len(loader)

    train_acc = 100. * correct / total

    return train_loss, train_acc


# =========================
# 验证
# =========================
@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()

    total = 0
    correct = 0
    loss_sum = 0

    preds = []
    targets = []

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

    val_loss = loss_sum / len(loader)

    val_acc = 100. * correct / total

    return val_loss, val_acc, preds, targets


# =========================
# 主函数
# =========================
def main():

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("Device:", device)

    # =========================
    # 加载数据
    # =========================

    # 小文件使用：
    # X, y, info = load_training_data_from_csv(
    #     './shuffled_dataset/train.csv'
    # )

    # 大文件使用：
    X, y, info = load_large_csv_in_chunks(
        './shuffled_dataset/train.csv',
        chunk_size=10000
    )

    # =========================
    # DataLoader
    # =========================
    train_loader, val_loader = create_dataloaders(
        X,
        y,
        batch_size=64,
        val_ratio=0.1
    )

    # =========================
    # 模型
    # =========================
    model = CNN1DClassifier(
        input_dim=4000,
        num_classes=info['n_classes']
    ).to(device)

    print(model)

    # =========================
    # Loss & Optimizer
    # =========================
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    scaler = torch.cuda.amp.GradScaler()

    # =========================
    # 保存训练过程
    # =========================
    train_losses = []
    train_accs = []

    val_losses = []
    val_accs = []

    best_acc = 0.0

    # =========================
    # 开始训练
    # =========================
    for epoch in range(1, 51):

        print(f"\n===== Epoch {epoch} =====")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            epoch
        )

        val_loss, val_acc, preds, targets = validate(
            model,
            val_loader,
            criterion,
            device
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}%"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # 保存最佳模型
        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                'best_1dcnn_model.pth'
            )

            print("✓ Saved best model")

    # =========================
    # 最终结果
    # =========================
    print("\n=========================")
    print("Final Report")
    print("=========================")

    print(classification_report(targets, preds))

    print("Confusion Matrix:")

    print(confusion_matrix(targets, preds))

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")

    # =========================
    # 绘制Loss曲线
    # =========================
    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.title('Training and Validation Loss')

    plt.legend()

    plt.grid(True)

    plt.savefig('loss_curve.png', dpi=300)

    plt.close()

    # =========================
    # 绘制Accuracy曲线
    # =========================
    plt.figure(figsize=(8, 5))

    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.title('Training and Validation Accuracy')

    plt.legend()

    plt.grid(True)

    plt.savefig('accuracy_curve.png', dpi=300)

    plt.close()

    # =========================
    # 保存训练记录
    # =========================
    history = pd.DataFrame({
        'epoch': np.arange(1, 51),
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    })

    history.to_csv(
        'training_history.csv',
        index=False
    )

    print("\n训练完成")
    print("已保存:")
    print("1. best_1dcnn_model.pth")
    print("2. loss_curve.png")
    print("3. accuracy_curve.png")
    print("4. training_history.csv")


if __name__ == "__main__":
    main()