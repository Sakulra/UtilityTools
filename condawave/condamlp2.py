import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from pathlib import Path
import pickle
import gc
import sys
import json
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# ==================== 系统检测 ====================
IS_WINDOWS = sys.platform.startswith('win')
print(f"Operating System: {'Windows' if IS_WINDOWS else 'Non-Windows'}")

# ==================== 配置参数（使用普通字典） ====================
CONFIG = {
    # 原始文件路径
    'csv_files' : [
        'F:\shiyan_data\processA\cft_processed_data.csv',
        'F:\shiyan_data\processA\qiu_processed_data.csv', 
        'F:\shiyan_data\processA\\tuoyuan_processed_data.csv',
        'F:\shiyan_data\processA\yuanzhu_processed_data.csv',
        'F:\shiyan_data\processA\zft_processed_data.csv'
    ],
    
    # 预处理保存路径
    'processed_dir': 'processed_data',
    'train_data_file': 'train_data.npy',
    'train_labels_file': 'train_labels.npy',
    'test_data_file': 'test_data.npy',
    'test_labels_file': 'test_labels.npy',
    'metadata_file': 'metadata.json',
    
    # 训练参数
    'batch_size': 512 if IS_WINDOWS else 2048,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'num_classes': 5,
    'hidden_layers': [512, 256, 128],
    
    # 数据分割
    'train_ratio': 0.8,
    
    # Windows特定的优化
    'num_workers': 0 if IS_WINDOWS else 4,
    
    # 设备设置
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    
    # 模型保存路径
    'model_save_path': 'mlp_model.pth',
    'config_save_path': 'training_config.json',
    
    # 混合精度训练
    'use_amp': True,
    
    # 梯度累积
    'gradient_accumulation_steps': 4 if IS_WINDOWS else 1,
    
    # 内存映射参数
    'mmap_mode': 'r',
}

# ==================== 数据预处理器 ====================
class DataPreprocessor:
    """数据预处理器"""
    
    @staticmethod
    def get_file_info(file_path: str) -> Tuple[int, int]:
        """获取文件信息"""
        try:
            # 读取第一行获取特征维度
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            
            n_columns = len(first_line.split(','))
            feature_dim = n_columns - 1  # 最后一列是标签
            
            # 统计行数
            print(f"  Counting lines in {os.path.basename(file_path)}...")
            line_count = 0
            with open(file_path, 'r') as f:
                next(f)  # 跳过标题行
                for line in f:
                    line_count += 1
            
            return line_count, feature_dim
            
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            raise
    
    @staticmethod
    def process_single_file(file_idx: int, file_path: str, output_dir: str) -> Dict[str, Any]:
        """处理单个文件"""
        print(f"\nProcessing file {file_idx+1}: {os.path.basename(file_path)}")
        
        total_samples, feature_dim = DataPreprocessor.get_file_info(file_path)
        train_size = int(total_samples * CONFIG['train_ratio'])
        
        # 创建临时文件
        temp_dir = os.path.join(output_dir, 'temp')
        Path(temp_dir).mkdir(exist_ok=True)
        
        train_file = os.path.join(temp_dir, f'train_{file_idx}.npy')
        train_labels_file = os.path.join(temp_dir, f'train_labels_{file_idx}.npy')
        test_file = os.path.join(temp_dir, f'test_{file_idx}.npy')
        test_labels_file = os.path.join(temp_dir, f'test_labels_{file_idx}.npy')
        
        # 处理训练数据
        print(f"  Processing training data ({train_size:,} samples)...")
        train_chunks = []
        train_label_chunks = []
        
        chunk_size = 50000
        n_chunks = (train_size + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            skip_rows = 1 + chunk_idx * chunk_size  # 1 for header
            nrows = min(chunk_size, train_size - chunk_idx * chunk_size)
            
            try:
                df = pd.read_csv(
                    file_path,
                    skiprows=skip_rows,
                    nrows=nrows,
                    header=None,
                    dtype=np.float32,
                    low_memory=False
                )
                
                features = df.iloc[:, :-1].values.astype(np.float32)
                labels = np.full(len(features), file_idx, dtype=np.int32)
                
                train_chunks.append(features)
                train_label_chunks.append(labels)
                
                print(f"    Chunk {chunk_idx+1}/{n_chunks}: {len(features):,} samples")
                gc.collect()
                
            except Exception as e:
                print(f"    Error in chunk {chunk_idx+1}: {e}")
                continue
        
        # 保存训练数据
        if train_chunks:
            train_data = np.concatenate(train_chunks, axis=0)
            train_labels = np.concatenate(train_label_chunks, axis=0)
            
            np.save(train_file, train_data)
            np.save(train_labels_file, train_labels)
            
            del train_data, train_labels
            gc.collect()
        
        # 处理测试数据
        test_size = total_samples - train_size
        print(f"  Processing test data ({test_size:,} samples)...")
        
        test_chunks = []
        test_label_chunks = []
        
        n_chunks = (test_size + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            skip_rows = 1 + train_size + chunk_idx * chunk_size
            nrows = min(chunk_size, test_size - chunk_idx * chunk_size)
            
            try:
                df = pd.read_csv(
                    file_path,
                    skiprows=skip_rows,
                    nrows=nrows,
                    header=None,
                    dtype=np.float32,
                    low_memory=False
                )
                
                features = df.iloc[:, :-1].values.astype(np.float32)
                labels = np.full(len(features), file_idx, dtype=np.int32)
                
                test_chunks.append(features)
                test_label_chunks.append(labels)
                
                print(f"    Chunk {chunk_idx+1}/{n_chunks}: {len(features):,} samples")
                gc.collect()
                
            except Exception as e:
                print(f"    Error in chunk {chunk_idx+1}: {e}")
                continue
        
        # 保存测试数据
        if test_chunks:
            test_data = np.concatenate(test_chunks, axis=0)
            test_labels = np.concatenate(test_label_chunks, axis=0)
            
            np.save(test_file, test_data)
            np.save(test_labels_file, test_labels)
            
            del test_data, test_labels
            gc.collect()
        
        return {
            'train_file': train_file,
            'train_labels_file': train_labels_file,
            'test_file': test_file,
            'test_labels_file': test_labels_file,
            'train_samples': train_size if train_chunks else 0,
            'test_samples': test_size if test_chunks else 0,
            'feature_dim': feature_dim
        }
    
    @staticmethod
    def merge_and_normalize(output_dir: str, file_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并并标准化数据"""
        print("\nMerging and normalizing data...")
        
        # 首先计算均值和标准差
        print("  Calculating mean and std from training data...")
        all_train_samples = []
        sample_limit = 200000  # 限制样本数以节省内存
        
        for info in file_infos:
            if info['train_samples'] > 0:
                train_data = np.load(info['train_file'], mmap_mode='r')
                n_samples = min(len(train_data), sample_limit // len(file_infos))
                indices = np.random.choice(len(train_data), n_samples, replace=False)
                all_train_samples.append(train_data[indices])
        
        if not all_train_samples:
            raise ValueError("No training data found!")
        
        sample_data = np.concatenate(all_train_samples, axis=0)
        mean = np.mean(sample_data, axis=0)
        std = np.std(sample_data, axis=0)
        std[std == 0] = 1.0
        
        del sample_data, all_train_samples
        gc.collect()
        
        # 处理训练数据
        print("  Processing training data...")
        train_data_chunks = []
        train_label_chunks = []
        
        for info in file_infos:
            if info['train_samples'] > 0:
                print(f"    Loading {os.path.basename(info['train_file'])}...")
                train_data = np.load(info['train_file'], mmap_mode='r')
                train_labels = np.load(info['train_labels_file'], mmap_mode='r')
                
                # 标准化
                train_data_normalized = (train_data - mean) / std
                
                train_data_chunks.append(train_data_normalized)
                train_label_chunks.append(train_labels)
        
        # 保存训练数据
        print("  Saving training data...")
        train_data_final = np.concatenate(train_data_chunks, axis=0)
        train_labels_final = np.concatenate(train_label_chunks, axis=0)
        
        # 打乱训练数据
        print("  Shuffling training data...")
        indices = np.random.permutation(len(train_data_final))
        train_data_final = train_data_final[indices]
        train_labels_final = train_labels_final[indices]
        
        np.save(os.path.join(output_dir, CONFIG['train_data_file']), train_data_final)
        np.save(os.path.join(output_dir, CONFIG['train_labels_file']), train_labels_final)
        
        del train_data_final, train_labels_final
        gc.collect()
        
        # 处理测试数据
        print("  Processing test data...")
        test_data_chunks = []
        test_label_chunks = []
        
        for info in file_infos:
            if info['test_samples'] > 0:
                print(f"    Loading {os.path.basename(info['test_file'])}...")
                test_data = np.load(info['test_file'], mmap_mode='r')
                test_labels = np.load(info['test_labels_file'], mmap_mode='r')
                
                # 标准化
                test_data_normalized = (test_data - mean) / std
                
                test_data_chunks.append(test_data_normalized)
                test_label_chunks.append(test_labels)
        
        # 保存测试数据
        print("  Saving test data...")
        test_data_final = np.concatenate(test_data_chunks, axis=0)
        test_labels_final = np.concatenate(test_label_chunks, axis=0)
        
        np.save(os.path.join(output_dir, CONFIG['test_data_file']), test_data_final)
        np.save(os.path.join(output_dir, CONFIG['test_labels_file']), test_labels_final)
        
        # 创建元数据
        metadata = {
            'feature_dim': file_infos[0]['feature_dim'],
            'mean': mean.tolist(),  # 转换为list以便JSON序列化
            'std': std.tolist(),
            'train_samples': sum(info['train_samples'] for info in file_infos),
            'test_samples': sum(info['test_samples'] for info in file_infos),
            'num_classes': CONFIG['num_classes']
        }
        
        # 保存元数据为JSON
        with open(os.path.join(output_dir, CONFIG['metadata_file']), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 清理临时文件
        print("\nCleaning up temporary files...")
        temp_dir = os.path.join(output_dir, 'temp')
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        
        print(f"\n✓ Data processing complete!")
        print(f"  Train samples: {metadata['train_samples']:,}")
        print(f"  Test samples: {metadata['test_samples']:,}")
        print(f"  Feature dimension: {metadata['feature_dim']}")
        
        return metadata
    
    @staticmethod
    def preprocess_all_data() -> Dict[str, Any]:
        """预处理所有数据"""
        print("=" * 60)
        print("Data Preprocessing")
        print("=" * 60)
        
        # 检查输入文件
        print("\nChecking input files...")
        for file_path in CONFIG['csv_files']:
            if os.path.exists(file_path):
                print(f"✓ {file_path}")
            else:
                raise FileNotFoundError(f"✗ {file_path} not found!")
        
        # 创建输出目录
        output_dir = CONFIG['processed_dir']
        Path(output_dir).mkdir(exist_ok=True)
        
        # 处理每个文件
        file_infos = []
        for file_idx, file_path in enumerate(CONFIG['csv_files']):
            info = DataPreprocessor.process_single_file(file_idx, file_path, output_dir)
            file_infos.append(info)
        
        # 合并和标准化
        metadata = DataPreprocessor.merge_and_normalize(output_dir, file_infos)
        
        return metadata

# ==================== 数据集类 ====================
class NPYDataset(Dataset):
    """加载.npy文件的数据集"""
    
    def __init__(self, data_path: str, labels_path: str, metadata: Dict[str, Any]):
        self.data_path = data_path
        self.labels_path = labels_path
        self.metadata = metadata
        
        # 使用内存映射加载
        print(f"Loading dataset: {os.path.basename(data_path)}")
        self.data = np.load(data_path, mmap_mode=CONFIG['mmap_mode'])
        self.labels = np.load(labels_path, mmap_mode=CONFIG['mmap_mode'])
        
        if len(self.data) != len(self.labels):
            raise ValueError(f"Data and labels have different lengths: {len(self.data)} vs {len(self.labels)}")
        
        self.num_samples = len(self.labels)
        
        print(f"  Samples: {self.num_samples:,}")
        print(f"  Features: {self.data.shape[1]}")
        print(f"  Memory mapping: {CONFIG['mmap_mode']}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        # 直接从内存映射文件读取
        features = self.data[idx].astype(np.float32).copy()  # 复制以避免内存映射问题
        label = int(self.labels[idx])
        
        return torch.FloatTensor(features), torch.LongTensor([label]).squeeze()

# ==================== 模型定义 ====================
class MLPModel(nn.Module):
    """多层感知机模型"""
    
    def __init__(self, input_size: int, num_classes: int, hidden_layers: List[int]):
        super(MLPModel, self).__init__()
        
        print(f"\nBuilding MLP Model:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden layers: {hidden_layers}")
        print(f"  Output classes: {num_classes}")
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.model = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.model:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# ==================== 训练工具类 ====================
class Trainer:
    """训练器类"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(enabled=CONFIG['use_amp'])
        
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer,
        gradient_accumulation_steps: int = 1
    ) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        optimizer.zero_grad()
        
        print("Training...")
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=CONFIG['use_amp']):
                output = self.model(data)
                loss = criterion(output, target) / gradient_accumulation_steps
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度累积
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
            
            # 统计
            running_loss += loss.item() * gradient_accumulation_steps
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 打印进度
            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                acc = 100. * correct / total
                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {avg_loss:.4f}, Acc: {acc:.2f}%, Time: {elapsed:.1f}s")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, test_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print("Validating...")
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if (batch_idx + 1) % 50 == 0:
                    acc = 100. * correct / total
                    print(f"  Batch {batch_idx+1}/{len(test_loader)} - Acc: {acc:.2f}%")
        
        epoch_loss = running_loss / len(test_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc

# ==================== 模型保存和加载 ====================
def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    trainer: Trainer,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    best_acc: float,
    history: Dict[str, List[float]],
    metadata: Dict[str, Any],
    filepath: str
):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': trainer.scaler.state_dict() if CONFIG['use_amp'] else None,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'best_acc': best_acc,
        'history': history,
        'metadata': metadata,
        'config': {
            'hidden_layers': CONFIG['hidden_layers'],
            'batch_size': CONFIG['batch_size'],
            'learning_rate': CONFIG['learning_rate'],
            'num_classes': CONFIG['num_classes'],
        }
    }
    
    torch.save(checkpoint, filepath)
    print(f"✓ Model saved to {filepath}")

def load_checkpoint(filepath: str, device: torch.device) -> Dict[str, Any]:
    """加载检查点"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint

# ==================== 主训练流程 ====================
def train_model():
    """主训练函数"""
    print("=" * 60)
    print("MLP Model Training")
    print("=" * 60)
    
    device = CONFIG['device']
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # 检查预处理数据
    processed_dir = CONFIG['processed_dir']
    required_files = [
        CONFIG['train_data_file'],
        CONFIG['train_labels_file'],
        CONFIG['test_data_file'],
        CONFIG['test_labels_file'],
        CONFIG['metadata_file']
    ]
    
    all_files_exist = all(os.path.exists(os.path.join(processed_dir, f)) for f in required_files)
    
    if not all_files_exist:
        print("\nProcessed data not found. Running preprocessing...")
        metadata = DataPreprocessor.preprocess_all_data()
    else:
        print("\nLoading existing processed data...")
        with open(os.path.join(processed_dir, CONFIG['metadata_file']), 'r') as f:
            metadata = json.load(f)
        
        # 转换均值和标准差回numpy数组
        metadata['mean'] = np.array(metadata['mean'])
        metadata['std'] = np.array(metadata['std'])
        
        print(f"✓ Loaded metadata:")
        print(f"  Train samples: {metadata['train_samples']:,}")
        print(f"  Test samples: {metadata['test_samples']:,}")
        print(f"  Feature dimension: {metadata['feature_dim']}")
    
    # 创建数据集
    print("\nCreating datasets...")
    train_dataset = NPYDataset(
        os.path.join(processed_dir, CONFIG['train_data_file']),
        os.path.join(processed_dir, CONFIG['train_labels_file']),
        metadata
    )
    
    test_dataset = NPYDataset(
        os.path.join(processed_dir, CONFIG['test_data_file']),
        os.path.join(processed_dir, CONFIG['test_labels_file']),
        metadata
    )
    
    # 创建数据加载器
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=device.type == 'cuda'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'] * 2,
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=device.type == 'cuda'
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 创建模型
    print("\nCreating model...")
    model = MLPModel(
        input_size=metadata['feature_dim'],
        num_classes=metadata['num_classes'],
        hidden_layers=CONFIG['hidden_layers']
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 保存配置
    with open(CONFIG['config_save_path'], 'w') as f:
        json.dump(CONFIG, f, indent=2, default=str)
    print(f"Configuration saved to {CONFIG['config_save_path']}")
    
    # 创建训练器
    trainer = Trainer(model, device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=1e-4
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['num_epochs']
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    best_acc = 0.0
    start_epoch = 0
    
    # 尝试加载之前的检查点
    if os.path.exists(CONFIG['model_save_path']):
        print(f"\nFound existing model at {CONFIG['model_save_path']}")
        try:
            checkpoint = load_checkpoint(CONFIG['model_save_path'], device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if CONFIG['use_amp'] and checkpoint['scaler_state_dict']:
                trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            best_acc = checkpoint['best_acc']
            history = checkpoint['history']
            start_epoch = checkpoint['epoch'] + 1
            
            print(f"Resuming from epoch {start_epoch}")
            print(f"Previous best accuracy: {best_acc:.2f}%")
            
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Starting fresh training...")
    
    # 训练循环
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        for epoch in range(start_epoch, CONFIG['num_epochs']):
            print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = trainer.train_epoch(
                train_loader, criterion, optimizer, CONFIG['gradient_accumulation_steps']
            )
            
            # 验证
            val_loss, val_acc = trainer.validate(test_loader, criterion)
            
            # 更新学习率
            scheduler.step()
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    trainer=trainer,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    best_acc=best_acc,
                    history=history,
                    metadata=metadata,
                    filepath=CONFIG['model_save_path']
                )
            
            # 打印结果
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Best Val Acc: {best_acc:.2f}%")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
                save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    trainer=trainer,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    best_acc=best_acc,
                    history=history,
                    metadata=metadata,
                    filepath=checkpoint_path
                )
            
            # 早停检查
            if epoch > 10:
                recent_acc = history['val_acc'][-5:]
                if val_acc < max(recent_acc) - 2.0:  # 准确率下降超过2%
                    print("Validation accuracy not improving, consider early stopping...")
                    if epoch > 20:  # 至少训练20个epoch
                        print("Early stopping triggered!")
                        break
            
            # 清理GPU内存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    
    # 最终评估
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    # 加载最佳模型进行评估
    if os.path.exists(CONFIG['model_save_path']):
        print("Loading best model for evaluation...")
        checkpoint = load_checkpoint(CONFIG['model_save_path'], device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        final_loss, final_acc = trainer.validate(test_loader, criterion)
        
        print(f"\nTraining Summary:")
        print(f"  Total Time: {total_time:.2f} seconds")
        print(f"  Total Epochs: {len(history['train_loss'])}")
        print(f"  Final Test Loss: {final_loss:.4f}")
        print(f"  Final Test Accuracy: {final_acc:.2f}%")
        print(f"  Best Accuracy: {best_acc:.2f}%")
        
        # 保存训练历史
        history_df = pd.DataFrame({
            'epoch': list(range(1, len(history['train_loss']) + 1)),
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_loss': history['val_loss'],
            'val_acc': history['val_acc'],
            'learning_rate': history['learning_rate']
        })
        
        history_df.to_csv('training_history.csv', index=False)
        print("Training history saved to 'training_history.csv'")
        
        # 绘制训练曲线（可选）
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Train Acc')
            plt.plot(history['val_acc'], label='Val Acc')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.title('Training and Validation Accuracy')
            
            plt.tight_layout()
            plt.savefig('training_curves.png', dpi=150)
            print("Training curves saved to 'training_curves.png'")
            
        except ImportError:
            print("Matplotlib not available, skipping plot generation")
    
    else:
        print("No model found for evaluation")
    
    return model, history

# ==================== 推理函数 ====================
def load_trained_model(model_path: str = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """加载训练好的模型"""
    if model_path is None:
        model_path = CONFIG['model_save_path']
    
    # 加载配置
    with open(CONFIG['config_save_path'], 'r') as f:
        config = json.load(f)
    
    # 加载元数据
    with open(os.path.join(CONFIG['processed_dir'], CONFIG['metadata_file']), 'r') as f:
        metadata = json.load(f)
    
    # 创建模型
    model = MLPModel(
        input_size=metadata['feature_dim'],
        num_classes=metadata['num_classes'],
        hidden_layers=config['hidden_layers']
    ).to(CONFIG['device'])
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=CONFIG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, metadata

def predict(model: nn.Module, features: np.ndarray, metadata: Dict[str, Any]) -> Tuple[int, np.ndarray]:
    """使用模型进行预测"""
    model.eval()
    
    # 标准化
    mean = np.array(metadata['mean'])
    std = np.array(metadata['std'])
    features_normalized = (features - mean) / std
    
    # 转换为tensor
    features_tensor = torch.FloatTensor(features_normalized).unsqueeze(0).to(CONFIG['device'])
    
    with torch.no_grad():
        output = model(features_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class, probabilities.cpu().numpy()[0]

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    print("Starting MLP Training Pipeline")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Windows特定设置
    if IS_WINDOWS:
        print("\nWindows system detected - applying compatibility settings")
        torch.multiprocessing.set_start_method('spawn', force=True)
    
    try:
        # 训练模型
        model, history = train_model()
        
        print("\n" + "=" * 60)
        print("Training Pipeline Completed Successfully!")
        print("=" * 60)
        
        # 示例：如何加载和使用训练好的模型
        print("\nExample usage for inference:")
        print("```python")
        print("# Load trained model")
        print("model, metadata = load_trained_model()")
        print("")
        print("# Make prediction")
        print("# Assuming you have features as numpy array")
        print("# features = np.array([...])  # shape: (feature_dim,)")
        print("# predicted_class, probabilities = predict(model, features, metadata)")
        print("```")
        
    except Exception as e:
        print(f"\nFatal error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # 清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()