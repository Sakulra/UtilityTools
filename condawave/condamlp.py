import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm

# 配置参数
class Config:
    # 文件路径
    csv_files = [
        'F:\shiyan_data\processA\cft_processed_data.csv',
        'F:\shiyan_data\processA\qiu_processed_data.csv', 
        'F:\shiyan_data\processA\\tuoyuan_processed_data.csv',
        'F:\shiyan_data\processA\yuanzhu_processed_data.csv',
        'F:\shiyan_data\processA\zft_processed_data.csv'
    ]
    
    # 训练参数
    batch_size = 1024
    learning_rate = 0.001
    num_epochs = 50
    num_classes = 5
    hidden_layers = [512, 256, 128]  # 隐藏层维度
    
    # 数据分割
    train_ratio = 0.8
    num_workers = 4  # 数据加载的进程数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 保存路径
    model_save_path = 'mlp_model.pth'

class StreamingCSVDataset(Dataset):
    """
    更高效的数据集类，只在需要时读取数据
    """
    def __init__(self, file_indices, is_train=True):
        self.file_indices = file_indices
        self.is_train = is_train
        self.samples = []  # 存储(文件索引, 行号)的元组
        
        # 预获取输入维度（从第一个文件的第一行）
        self.input_size = self.get_input_size()
        
        # 收集所有样本的位置信息
        self.collect_sample_indices()
        
        # 缓存少量数据
        self.cache = {}
        self.cache_size = 10000
        
        print(f"Dataset created: {'Train' if is_train else 'Test'}")
        print(f"Number of samples: {len(self.samples)}")
        print(f"Input feature dimension: {self.input_size}")
    
    def get_input_size(self):
        """获取输入特征的维度"""
        # 从第一个文件的第一行获取特征维度
        file_path = Config.csv_files[0]
        
        # 读取第一行（跳过标题行）
        df = pd.read_csv(
            file_path,
            nrows=1,
            header=None
        )
        
        # 最后一列是标签，所以特征维度是列数减1
        input_size = df.shape[1] - 1
        return input_size
    
    def collect_sample_indices(self):
        """收集所有样本的位置信息而不加载数据"""
        print("Collecting sample indices...")
        
        for file_idx in self.file_indices:
            file_path = Config.csv_files[file_idx]
            
            try:
                # 获取文件总行数（包括标题行）
                with open(file_path, 'r') as f:
                    total_lines = sum(1 for _ in f)
                
                total_rows = total_lines - 1  # 减1排除标题行
                train_size = int(total_rows * Config.train_ratio)
                
                if self.is_train:
                    # 训练集：0 到 train_size-1
                    start, end = 0, train_size
                else:
                    # 测试集：train_size 到 total_rows-1
                    start, end = train_size, total_rows
                
                for row_idx in range(start, end):
                    self.samples.append((file_idx, row_idx))
                    
                print(f"  File {file_idx}: {end-start} samples")
                    
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
    
    def load_sample(self, file_idx, row_idx):
        """加载单个样本"""
        cache_key = (file_idx, row_idx)
        
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        file_path = Config.csv_files[file_idx]
        
        try:
            # 读取特定行
            df = pd.read_csv(
                file_path,
                skiprows=row_idx + 1,  # +1 跳过标题行
                nrows=1,
                header=None
            )
            
            # 分离特征和标签
            features = df.iloc[:, :-1].values[0].astype(np.float32)
            label = file_idx  # 文件索引即类别
            
            # 更新缓存
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = (features, label)
            
            return features, label
            
        except Exception as e:
            # 如果读取失败，返回零向量
            print(f"Error loading sample ({file_idx}, {row_idx}): {e}")
            features = np.zeros(self.input_size, dtype=np.float32)
            label = file_idx
            return features, label
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_idx, row_idx = self.samples[idx]
        features, label = self.load_sample(file_idx, row_idx)
        
        return torch.FloatTensor(features), torch.LongTensor([label]).squeeze()

class MLP(nn.Module):
    """多层感知机模型"""
    def __init__(self, input_size, num_classes, hidden_layers=[512, 256, 128]):
        super(MLP, self).__init__()
        
        print(f"Building MLP with input size: {input_size}")
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for i, hidden_size in enumerate(hidden_layers):
            print(f"  Layer {i+1}: {prev_size} -> {hidden_size}")
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        # 输出层
        print(f"  Output layer: {prev_size} -> {num_classes}")
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, test_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Validation')
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 收集预测结果用于后续分析
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            progress_bar.set_postfix({
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    
    # 计算每个类别的准确率
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_targets, all_predictions)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print("\nClass-wise Accuracy:")
    for i, acc in enumerate(class_acc):
        print(f"  Class {i}: {acc:.2%}")
    
    return epoch_loss, epoch_acc

def main():
    print("=" * 60)
    print("Multi-Layer Perceptron Training")
    print("=" * 60)
    print(f"Using device: {Config.device}")
    
    # 1. 首先创建训练数据集来获取输入维度
    print("\n1. Creating training dataset...")
    train_dataset = StreamingCSVDataset(
        file_indices=list(range(5)),  # 所有5个文件
        is_train=True
    )
    
    # 2. 创建测试数据集
    print("\n2. Creating test dataset...")
    test_dataset = StreamingCSVDataset(
        file_indices=list(range(5)),  # 所有5个文件
        is_train=False
    )
    
    # 3. 创建模型
    print("\n3. Creating MLP model...")
    model = MLP(
        input_size=train_dataset.input_size,
        num_classes=Config.num_classes,
        hidden_layers=Config.hidden_layers
    ).to(Config.device)
    
    # 打印模型信息
    print("\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 4. 创建数据加载器
    print("\n4. Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=min(Config.num_workers, os.cpu_count() // 2),
        pin_memory=True if Config.device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=min(Config.num_workers, os.cpu_count() // 2),
        pin_memory=True if Config.device.type == 'cuda' else False
    )
    
    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), 
                          lr=Config.learning_rate,
                          weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.num_epochs
    )
    
    # 6. 训练循环
    print("\n5. Starting training...")
    print("=" * 60)
    
    best_acc = 0.0
    train_history = []
    val_history = []
    
    for epoch in range(Config.num_epochs):
        print(f"\nEpoch {epoch+1}/{Config.num_epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.device
        )
        
        # 验证
        val_loss, val_acc = validate(
            model, test_loader, criterion, Config.device
        )
        
        # 更新学习率
        scheduler.step()
        
        # 保存历史记录
        train_history.append((train_loss, train_acc))
        val_history.append((val_loss, val_acc))
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_acc': best_acc,
                'input_size': train_dataset.input_size,
                'hidden_layers': Config.hidden_layers,
                'num_classes': Config.num_classes,
                'train_history': train_history,
                'val_history': val_history,
            }, Config.model_save_path)
            print(f"✓ Model saved with accuracy: {val_acc:.2f}%")
        
        # 打印epoch结果
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    
    # 7. 加载最佳模型并最终测试
    print("\n" + "=" * 60)
    print("6. Loading best model for final evaluation...")
    
    if os.path.exists(Config.model_save_path):
        checkpoint = torch.load(Config.model_save_path, map_location=Config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
        print(f"Best accuracy: {checkpoint['best_acc']:.2f}%")
    else:
        print("No saved model found, using the last model")
    
    # 最终评估
    print("\nFinal Evaluation:")
    print("-" * 50)
    final_loss, final_acc = validate(model, test_loader, criterion, Config.device)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Test Results:")
    print(f"  Test Loss: {final_loss:.4f}")
    print(f"  Test Accuracy: {final_acc:.2f}%")
    print(f"  Best Accuracy: {best_acc:.2f}%")
    print(f"  Model saved to: {Config.model_save_path}")
    print("=" * 60)

def test_inference():
    """测试推理代码"""
    print("\nTesting inference on single sample...")
    
    # 加载模型
    checkpoint = torch.load(Config.model_save_path, map_location=Config.device)
    
    # 重新创建模型架构
    model = MLP(
        input_size=checkpoint['input_size'],
        num_classes=checkpoint['num_classes'],
        hidden_layers=checkpoint['hidden_layers']
    ).to(Config.device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 创建一个随机输入样本
    input_size = checkpoint['input_size']
    test_input = torch.randn(1, input_size).to(Config.device)
    
    with torch.no_grad():
        output = model(test_input)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        
        print(f"\nInput shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Predicted class: {predicted_class.item()}")
        print(f"Class probabilities: {probabilities.cpu().numpy()}")

if __name__ == "__main__":
    # 检查文件是否存在
    print("Checking CSV files...")
    for file_path in Config.csv_files:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
    
    # 运行主训练流程
    main()
    
    # 可选：运行推理测试
    if os.path.exists(Config.model_save_path):
        test_inference()