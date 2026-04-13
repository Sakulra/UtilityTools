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

@torch.no_grad()
def inference_on_test(model_path, test_csv_path, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # =========================
    # 1. 加载测试数据
    # =========================
    print(">>> Loading test data...")
    df = pd.read_csv(test_csv_path)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    # 标签转0-based
    y = y - 1

    print("Test shape:", X.shape)

    test_ds = LargeNumpyDataset(X, y)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # =========================
    # 2. 加载模型
    # =========================
    num_classes = len(np.unique(y))

    model = CNNLSTMClassifier(
        input_dim=X.shape[1],
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # =========================
    # 3. 推理
    # =========================
    preds = []
    targets = []

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        pred = out.argmax(1)

        preds.extend(pred.cpu().numpy())
        targets.extend(y.cpu().numpy())

    # =========================
    # 4. 输出结果
    # =========================
    print("\nClassification Report:")
    print(classification_report(targets, preds))

    cm = confusion_matrix(targets, preds)
    print("Confusion Matrix:\n", cm)

    # =========================
    # 5. 画混淆矩阵（不用seaborn）
    # =========================
    plot_confusion_matrix(cm)

    return cm

#绘制混淆矩阵
def plot_confusion_matrix(cm, classes=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix", fontsize=16)
    plt.colorbar()

    num_classes = cm.shape[0]
    tick_marks = np.arange(num_classes)

    if classes is None:
        classes = [str(i) for i in range(num_classes)]

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # 在格子里写数值
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 推理测试集
    inference_on_test(
        model_path='cnnlstmfinal_best_model.pth',
        test_csv_path='./shuffled_dataset/train.csv',
        batch_size=64
    )