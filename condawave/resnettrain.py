import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
import os
import json
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class SpectrogramDataset(Dataset):
    """时频图像数据集"""
    def __init__(self, data_path, transform=None, split='train'):
        """
        参数:
        - data_path: 数据路径
        - transform: 数据增强变换
        - split: 'train', 'val', 或 'test'
        """
        self.data_path = data_path
        self.transform = transform
        self.split = split
        
        # 加载数据
        data = np.load(os.path.join(data_path, f'{split}_data.npz'))
        self.images = data['images']
        self.labels = data['labels']
        
        print(f"加载 {split} 数据集: {len(self.images)} 个样本")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 转换为PyTorch tensor (C, H, W)
        image = torch.FloatTensor(image.transpose(2, 0, 1))
        
        # 数据增强
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ResNetClassifier(nn.Module):
    """基于ResNet的分类器"""
    def __init__(self, num_classes, backbone='resnet50', pretrained=True):
        super(ResNetClassifier, self).__init__()
        
        # 加载预训练模型
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            in_features = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            in_features = 2048
        else:
            raise ValueError(f"不支持的backbone: {backbone}")
        
        # 修改第一层以接受单通道输入
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            3, original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        # 复制预训练权重的平均值到所有通道
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight[:, 0] = original_conv1.weight[:, 0].mean(dim=1, keepdim=True).squeeze()
                self.backbone.conv1.weight[:, 1] = original_conv1.weight[:, 0].mean(dim=1, keepdim=True).squeeze()
                self.backbone.conv1.weight[:, 2] = original_conv1.weight[:, 0].mean(dim=1, keepdim=True).squeeze()
        
        # 修改最后的全连接层
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        # 冻结前面几层（可选）
        self._freeze_layers()
    
    def _freeze_layers(self, freeze_until=5):
        """冻结前面几层，只训练后面的层"""
        layer_counter = 0
        
        # 冻结卷积层
        for name, param in self.backbone.named_parameters():
            if 'layer' in name:
                layer_num = int(name.split('.')[1])
                if layer_num < freeze_until:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif 'conv1' in name or 'bn1' in name:
                param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)

class EfficientNetClassifier(nn.Module):
    """基于EfficientNet的分类器"""
    def __init__(self, num_classes, model_name='efficientnet_b0', pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        
        from efficientnet_pytorch import EfficientNet
        
        # 加载预训练模型
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            self.backbone = EfficientNet.from_name(model_name)
        
        # 获取特征维度
        in_features = self.backbone._fc.in_features
        
        # 修改最后的全连接层
        self.backbone._fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def get_data_transforms(img_size=224):
    """获取数据增强变换"""
    # 训练集的数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集和测试集的数据增强（只有标准化）
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_data_loaders(data_dir, batch_size, img_size=224):
    """创建数据加载器"""
    train_transform, val_transform = get_data_transforms(img_size)
    
    # 创建数据集
    train_dataset = SpectrogramDataset(data_dir, train_transform, 'train')
    val_dataset = SpectrogramDataset(data_dir, val_transform, 'val')
    test_dataset = SpectrogramDataset(data_dir, val_transform, 'test')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")
    print(f"测试集: {len(test_dataset)} 个样本")
    
    return train_loader, val_loader, test_loader

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scheduler=None):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 学习率调度（如果提供）
        if scheduler and hasattr(scheduler, 'batch_step'):
            scheduler.batch_step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, split='Val'):
    """验证/测试模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'[{split}]')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc, all_preds, all_targets

def plot_training_history(history, save_path='training_history.png'):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Validation Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Validation Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    return cm

def main():
    # 设置参数
    data_dir = 'spectrogram_data'
    model_name = 'resnet50'  # 可选: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'efficientnet_b0'
    pretrained = True
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    weight_decay = 1e-4
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 加载数据信息
    with open(os.path.join(data_dir, 'data_info.json'), 'r') as f:
        info = json.load(f)
    
    num_classes = info['n_classes']
    class_names = info['class_names']
    img_size = tuple(info['img_size'])
    
    print(f"\n数据信息:")
    print(f"类别数: {num_classes}")
    print(f"图像尺寸: {img_size}")
    print(f"训练样本: {info['n_train']}")
    print(f"验证样本: {info['n_val']}")
    print(f"测试样本: {info['n_test']}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, batch_size, img_size[0]
    )
    
    # 创建模型
    if model_name.startswith('efficientnet'):
        model = EfficientNetClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained
        )
    else:
        model = ResNetClassifier(
            num_classes=num_classes,
            backbone=model_name,
            pretrained=pretrained
        )
    
    model = model.to(device)
    
    print(f"\n模型结构: {model_name}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = f'best_{model_name}.pth'
    
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device, 'Val'
        )
        
        # 更新学习率
        scheduler.step()
        
        # 保存训练历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\n训练结果:")
        print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'info': info
            }, best_model_path)
            print(f"✅ 保存最佳模型，验证准确率: {val_acc:.2f}%")
        
        # 每10个epoch保存一次检查点
        if epoch % 10 == 0:
            checkpoint_path = f'checkpoint_{model_name}_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"保存检查点到 {checkpoint_path}")
        
        # 早停（如果连续5个epoch验证损失没有改善）
        if epoch > 10 and val_loss > min(history['val_loss'][-5:]):
            print("⚠️ 验证损失连续5个epoch没有改善，考虑早停")
    
    end_time = time.time()
    print(f"\n训练完成！总时间: {end_time - start_time:.2f} 秒")
    
    # 绘制训练曲线
    plot_training_history(history, f'training_history_{model_name}.png')
    
    # 加载最佳模型进行最终测试
    print(f"\n加载最佳模型进行测试...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 在测试集上评估
    test_loss, test_acc, test_preds, test_targets = validate(
        model, test_loader, criterion, device, 'Test'
    )
    
    print(f"\n📊 最终测试结果:")
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.2f}%")
    
    # 保存最终模型
    final_model_path = f'final_{model_name}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'info': info,
        'test_acc': test_acc,
        'val_acc': best_val_acc
    }, final_model_path)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(test_targets, test_preds, class_names, 
                         f'confusion_matrix_{model_name}.png')
    
    # 打印分类报告
    print("\n📋 分类报告:")
    print(classification_report(
        test_targets, test_preds,
        target_names=class_names,
        digits=4
    ))
    
    # 保存结果
    results = {
        'model_name': model_name,
        'test_accuracy': float(test_acc),
        'best_val_accuracy': float(best_val_acc),
        'num_parameters': int(trainable_params),
        'training_time': end_time - start_time,
        'confusion_matrix': confusion_matrix(test_targets, test_preds).tolist(),
        'classification_report': classification_report(
            test_targets, test_preds, target_names=class_names, output_dict=True
        )
    }
    
    import json
    with open(f'results_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 所有模型已保存:")
    print(f"- 最佳模型: {best_model_path}")
    print(f"- 最终模型: {final_model_path}")
    print(f"- 训练历史图: training_history_{model_name}.png")
    print(f"- 混淆矩阵图: confusion_matrix_{model_name}.png")
    print(f"- 结果文件: results_{model_name}.json")
    
    # 可视化一些预测结果
    visualize_predictions(model, test_loader, device, class_names, num_samples=6)

def visualize_predictions(model, dataloader, device, class_names, num_samples=6):
    """可视化预测结果"""
    model.eval()
    
    # 获取一批数据
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # 预测
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    # 转换回CPU
    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()
    probs = probs.cpu()
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        ax = axes[i]
        
        # 显示图像
        img = images[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0, 1]
        
        ax.imshow(img[:, :, 0], cmap='viridis')
        
        # 设置标题
        true_label = class_names[labels[i].item()]
        pred_label = class_names[preds[i].item()]
        confidence = probs[i][preds[i]].item() * 100
        
        color = 'green' if labels[i] == preds[i] else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%", 
                    color=color, fontsize=10)
        
        ax.axis('off')
    
    plt.suptitle("模型预测示例 (绿色: 正确, 红色: 错误)", fontsize=12)
    plt.tight_layout()
    plt.savefig('prediction_examples.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()