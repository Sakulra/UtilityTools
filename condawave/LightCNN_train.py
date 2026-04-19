import torch
from torch.utils.data import Dataset
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import csv

class AcousticDataset(Dataset):
    def __init__(self, data_dir, is_training=True):
        self.files = []
        self.labels = []
        self.is_training = is_training

        for label in range(1, 6):
            class_dir = os.path.join(data_dir, str(label))
            if os.path.exists(class_dir):
                for f in os.listdir(class_dir):
                    if f.endswith(".npy"):
                        self.files.append(os.path.join(class_dir, f))
                        self.labels.append(label - 1)

    def __len__(self):
        return len(self.files)

    def spec_augment(self, x):
        """SpecAugment: 时间mask + 频率mask"""
        _, H, W = x.shape

        # 时间mask
        if np.random.rand() < 0.5:
            t = np.random.randint(0, W // 4)
            t0 = np.random.randint(0, W - t)
            x[:, :, t0:t0+t] = 0

        # 频率mask
        if np.random.rand() < 0.5:
            f = np.random.randint(0, H // 4)
            f0 = np.random.randint(0, H - f)
            x[:, f0:f0+f, :] = 0

        return x

    def __getitem__(self, idx):
        feature = np.load(self.files[idx])  # (H, W)

        # 转 tensor + 单通道
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)

        # 标准化（每张图）
        feature = (feature - feature.mean()) / (feature.std() + 1e-6)

        # 数据增强（仅训练）
        if self.is_training:
            if np.random.rand() < 0.5:
                feature = torch.flip(feature, dims=[2])  # 时间翻转

            feature = self.spec_augment(feature)

        label = self.labels[idx]
        return feature, label
    


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 52x59 → 26x29

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 13x14

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

#训练代码

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), 100 * correct / total


def main():
    data_dir = "F:/shiyan_data/dataset"
    # data_dir = "./dataset"
    batch_size = 64
    epochs = 30
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    log_filename = "training_results.csv"
    with open(log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    train_dataset = AcousticDataset(os.path.join(data_dir, "train"), True)
    val_dataset = AcousticDataset(os.path.join(data_dir, "val"), False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN().to(device)

    # 类别权重（防不平衡）
    labels = np.array(train_dataset.labels)
    class_counts = np.bincount(labels)
    weights = 1. / (class_counts + 1e-6)
    weights = weights / weights.sum() * len(class_counts)
    weights = torch.FloatTensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "shipinsimple_best_model.pth")
            print("✓ 保存最佳模型")
        with open(log_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])

    print("Best Acc:", best_acc)


if __name__ == "__main__":
    main()