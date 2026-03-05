import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import argparse
import warnings
warnings.filterwarnings('ignore')

# 1. 首先确保数据已转换
# python transfer.py --csv_path your_data.csv --output_dir ./dataset

# 2. 运行单个模型测试
# python train.py --data_dir ./dataset --model_type resnet18 --epochs 50
# python train.py --data_dir ./dataset --model_type resnet50 --epochs 50
# python train.py --data_dir ./dataset --model_type improved --epochs 50

# 3. 运行完整对比测试
# python compare_models.py

class AcousticDataset(Dataset):
    """声波图像数据集"""
    def __init__(self, data_dir, transform=None, is_training=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_training = is_training
        self.images = []
        self.labels = []
        
        # 遍历每个类别文件夹
        class_names = ['长方体', '球体', '椭圆', '圆柱体', '正方体']
        for label in range(1, 6):
            class_dir = os.path.join(data_dir, str(label))
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(label - 1)  # 转换为0-4的标签
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"Error loading image: {img_path}")
            # 返回一个随机样本
            return self.__getitem__(np.random.randint(0, len(self.images)))
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class AcousticAttentionBlock(nn.Module):
    """注意力机制模块 - 帮助网络关注重要的时频区域"""
    def __init__(self, channels):
        super(AcousticAttentionBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # 空间注意力
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x_out = x_ca * sa
        
        return x_out + x  # 残差连接

class FrequencyFeatureExtractor(nn.Module):
    """专门提取频率特征的模块"""
    def __init__(self, in_channels, out_channels):
        super(FrequencyFeatureExtractor, self).__init__()
        
        # 不同尺度的卷积核捕捉不同频率特征
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, (3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels, out_channels//4, (5, 1), padding=(2, 0))
        self.conv3 = nn.Conv2d(in_channels, out_channels//4, (7, 1), padding=(3, 0))
        self.conv4 = nn.Conv2d(in_channels, out_channels//4, (1, 3), padding=(0, 1))
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        return self.relu(self.bn(x_cat))

class TemporalFeatureExtractor(nn.Module):
    """专门提取时间特征的模块"""
    def __init__(self, in_channels, out_channels):
        super(TemporalFeatureExtractor, self).__init__()
        
        # 双向LSTM处理时间序列
        self.lstm = nn.LSTM(in_channels, out_channels//2, 
                            bidirectional=True, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        batch, channels, height, width = x.shape
        
        # 重排为 (batch, width, channels*height) 以处理时间维度
        x_reshaped = x.permute(0, 3, 1, 2).contiguous()
        x_reshaped = x_reshaped.view(batch, width, -1)
        
        lstm_out, _ = self.lstm(x_reshaped)
        lstm_out = lstm_out.permute(0, 2, 1).contiguous()
        lstm_out = lstm_out.view(batch, -1, height, width)
        
        return lstm_out

class ImprovedAcousticCNN(nn.Module):
    """改进的声波分类CNN"""
    def __init__(self, num_classes=5, input_channels=3):
        super(ImprovedAcousticCNN, self).__init__()
        
        # 初始特征提取
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # 频率特征提取分支
        self.freq_branch = nn.Sequential(
            FrequencyFeatureExtractor(32, 64),
            nn.MaxPool2d(2),
            FrequencyFeatureExtractor(64, 128),
        )
        
        # 时间特征提取分支
        self.temp_branch = nn.Sequential(
            TemporalFeatureExtractor(32, 64),
            nn.MaxPool2d(2),
            TemporalFeatureExtractor(64, 128),
        )
        
        # 注意力模块
        self.attention1 = AcousticAttentionBlock(128)
        self.attention2 = AcousticAttentionBlock(256)
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # 辅助分类器（用于多任务学习）
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x, return_features=False):
        # 初始特征提取
        x = self.initial_conv(x)
        
        # 分支处理
        freq_feat = self.freq_branch(x)
        temp_feat = self.temp_branch(x)
        
        # 特征融合
        combined = torch.cat([freq_feat, temp_feat], dim=1)
        combined = self.attention1(combined)
        combined = self.attention2(combined)
        
        # 全局特征
        global_feat = self.fusion(combined)
        
        # 分类
        output = self.classifier(global_feat)
        
        if return_features:
            # 返回特征用于可视化
            return output, global_feat
        return output

class LightweightAcousticCNN(nn.Module):
    """轻量级声波分类CNN（适用于小数据集）"""
    def __init__(self, num_classes=5):
        super(LightweightAcousticCNN, self).__init__()
        
        self.features = nn.Sequential(
            # 块1
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # 块2
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # 块3
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def mixup_data(x, y, alpha=1.0):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True):
    """训练一个epoch（支持Mixup）"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        
        if use_mixup and np.random.random() > 0.5:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            # 对于准确率，使用原始标签
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(labels_a).sum().item() + 
                       (1 - lam) * predicted.eq(labels_b).sum().item())
            total += labels.size(0)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_features = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if hasattr(model, 'forward') and 'return_features' in model.forward.__code__.co_varnames:
                outputs, features = model(inputs, return_features=True)
                all_features.append(features.cpu().numpy())
            else:
                outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return (running_loss / len(dataloader), 
            100. * correct / total, 
            all_preds, 
            all_labels,
            np.concatenate(all_features) if all_features else None)

def plot_feature_distribution(features, labels, class_names):
    """使用t-SNE可视化特征分布"""
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   label=class_names[i], alpha=0.7)
    
    plt.title('Feature Distribution (t-SNE)')
    plt.legend()
    plt.savefig('feature_distribution.png')
    plt.show()

def get_class_weights(dataset):
    """计算类别权重用于处理不平衡"""
    labels = np.array([label for _, label in dataset])
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    return torch.FloatTensor(class_weights)

def main():
    parser = argparse.ArgumentParser(description='声波图像分类训练（改进版）')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--model_type', type=str, default='improved',
                        choices=['improved', 'lightweight', 'resnet18', 'resnet50'],
                        help='模型类型')
    parser.add_argument('--use_focal_loss', action='store_true', help='使用Focal Loss')
    parser.add_argument('--use_mixup', action='store_true', help='使用Mixup增强')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据预处理（针对声波图像的专门增强）
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),  # 时间轴翻转
        transforms.RandomVerticalFlip(p=0.1),    # 频率轴翻转（谨慎使用）
        transforms.RandomRotation(5),             # 小角度旋转
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.05),  # 平移增强
            scale=(0.9, 1.1)         # 缩放增强
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    train_dataset = AcousticDataset(train_dir, transform=train_transform, is_training=True)
    val_dataset = AcousticDataset(val_dir, transform=val_transform, is_training=False)
    
    # 处理类别不平衡
    class_weights = get_class_weights(train_dataset)
    print(f"类别权重: {class_weights}")
    
    # 创建数据加载器（使用加权采样处理不平衡）
    sampler = WeightedRandomSampler(
        weights=class_weights[train_dataset.labels],
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    if args.model_type == 'improved':
        print("使用改进的声波专用CNN模型...")
        model = ImprovedAcousticCNN(num_classes=5)
    elif args.model_type == 'lightweight':
        print("使用轻量级CNN模型...")
        model = LightweightAcousticCNN(num_classes=5)
    elif args.model_type == 'resnet18':
        print("使用ResNet18模型...")
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, 5)
    elif args.model_type == 'resnet50':
        print("使用ResNet50模型...")
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, 5)
    
    model = model.to(device)
    
    # 损失函数
    if args.use_focal_loss:
        print("使用Focal Loss...")
        criterion = FocalLoss(alpha=class_weights.to(device), gamma=2)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # 早停机制
    early_stopping_patience = 15
    early_stopping_counter = 0
    best_val_acc = 0.0
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    print("开始训练...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            use_mixup=args.use_mixup
        )
        
        # 验证
        val_loss, val_acc, val_preds, val_labels, features = validate(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'best_model.pth')
            print(f"✓ 保存最佳模型，验证准确率: {best_val_acc:.2f}%")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"早停: {early_stopping_patience}轮未改善")
                break
    
    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 加载最佳模型进行最终评估
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    _, _, final_preds, final_labels, final_features = validate(
        model, val_loader, criterion, device
    )
    
    # 分类报告
    class_names = ['长方体', '球体', '椭圆', '圆柱体', '正方体']
    print("\n分类报告:")
    print(classification_report(final_labels, final_preds, 
                               target_names=class_names))
    
    # 混淆矩阵
    plot_confusion_matrix(final_labels, final_preds, class_names)
    
    # 特征可视化（如果模型返回特征）
    if final_features is not None:
        plot_feature_distribution(final_features, final_labels, class_names)

def plot_training_history(history):
    """绘制训练历史"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 损失曲线
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 学习率曲线
    ax3.plot(history['lr'], linewidth=2, color='green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)
    
    # 过拟合检测
    train_val_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    ax4.plot(train_val_gap, linewidth=2, color='red')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Train-Val Gap (%)')
    ax4.set_title('Overfitting Detection')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绝对数量混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix (Counts)')
    
    # 归一化混淆矩阵
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()