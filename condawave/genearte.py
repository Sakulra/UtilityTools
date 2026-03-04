import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json
import matplotlib

# 设置新罗马字体
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 18

# 设置随机种子保证可重复性
np.random.seed(40)

def generate_training_data(n_epochs=50):
    """
    生成模拟的ResNet50训练数据
    数据量：60800条
    目标准确率：94%
    """
    
    # 生成训练损失
    train_loss = []
    
    # 生成训练准确率
    train_acc = []
    
    for epoch in range(1, n_epochs + 1):
        # 训练损失：从2.8逐渐下降到0.15，加入随机波动
        if epoch <= 8:
            t_loss = 2.8 - 0.3 * epoch + np.random.normal(0, 0.08)
        elif epoch <= 20:
            t_loss = 0.8 - 0.04 * (epoch - 8) + np.random.normal(0, 0.05)
        elif epoch <= 35:
            t_loss = 0.35 - 0.008 * (epoch - 20) + np.random.normal(0, 0.03)
        else:
            t_loss = 0.18 - 0.001 * (epoch - 35) + np.random.normal(0, 0.02)
        
        # 确保损失为正
        t_loss = max(0.05, t_loss)
        
        train_loss.append(round(t_loss, 4))
        
        # 训练准确率：从38%逐渐上升到97%
        if epoch <= 8:
            t_acc = 38 + 5.5 * epoch + np.random.normal(0, 1.2)
        elif epoch <= 20:
            t_acc = 82 + 1.1 * (epoch - 8) + np.random.normal(0, 0.7)
        elif epoch <= 35:
            t_acc = 98 + 0.1 * (epoch - 20) + np.random.normal(0, 0.4)
        else:
            t_acc = 97.8 + np.random.normal(0, 0.2)
        
        # 确保准确率不超过100%
        t_acc = min(98, t_acc)
        
        train_acc.append(round(t_acc, 2))
    
    return train_loss, train_acc

def generate_confusion_matrix(n_samples=60800, accuracy=0.9782, n_classes=5):
    """
    生成更真实的混淆矩阵
    总样本数：60800
    准确率：94.12%
    5个类别
    """
    # 每个类别的真实样本数（略微不均匀，更符合实际情况）
    class_distribution = [0.22, 0.19, 0.21, 0.18, 0.20]  # 各类别比例
    true_counts = np.array([int(n_samples * p) for p in class_distribution])
    true_counts[-1] += n_samples - true_counts.sum()  # 调整最后一类使总和正确
    
    # 初始化混淆矩阵
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # 每个类别的错误分布模式（更真实的错误模式）
    error_patterns = {
        # 每个类更容易与哪些类混淆
        0: [1, 2, 3, 4],  # 类别0容易与类别1和2混淆
        1: [0, 2, 3, 4],  # 类别1容易与类别0和3混淆
        2: [0, 1, 3, 4],  # 类别2容易与类别0和4混淆
        3: [1, 2, 4, 0],  # 类别3容易与类别1和2混淆
        4: [2, 3, 0, 1]   # 类别4容易与类别2和3混淆
    }
    
    # 错误权重分布
    error_weights = {
        0: [0.45, 0.30, 0.15, 0.10],  # 错误分布权重
        1: [0.40, 0.35, 0.15, 0.10],
        2: [0.35, 0.30, 0.20, 0.15],
        3: [0.40, 0.30, 0.20, 0.10],
        4: [0.35, 0.30, 0.25, 0.10]
    }
    
    # 填充混淆矩阵
    for i in range(n_classes):
        # 正确分类的数量
        correct = int(true_counts[i] * accuracy)
        cm[i, i] = correct
        
        # 错误分类的数量
        errors = true_counts[i] - correct
        
        # 根据权重分配错误
        error_classes = error_patterns[i]
        weights = error_weights[i]
        
        for j, err_class in enumerate(error_classes):
            if j == len(error_classes) - 1:
                # 最后一个类别接收剩余的误差
                cm[i, err_class] = errors - sum(cm[i, ec] for ec in error_classes[:j])
            else:
                cm[i, err_class] = int(errors * weights[j])
    
    return cm, true_counts

def plot_training_curve(epochs, train_loss, train_acc):
    """绘制训练曲线（单y轴）"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 绘制损失曲线（左y轴）
    color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=18)
    ax1.set_ylabel('Loss', color=color, fontsize=18)
    line1 = ax1.plot(epochs, train_loss, 'r-', label='Training Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 3.5)
    ax1.grid(True, alpha=0.3)
    
    # 创建右y轴绘制准确率
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color, fontsize=12)
    line2 = ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    ax2.axhline(y=94, color='green', linestyle='--', alpha=0.7, label='94% Target')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(30, 100)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    # 添加目标线到图例
    lines.append(plt.Line2D([0], [0], color='green', linestyle='--', alpha=0.7))
    labels.append('94% Target')
    
    ax1.legend(lines, labels, loc='center right', fontsize=10)
    
    plt.title('ResNet50 Training Progress (50 Epochs)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150)
    plt.show()

def plot_confusion_matrix(cm, class_names, total_samples):
    """绘制混淆矩阵（不使用seaborn）"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    n_classes = len(class_names)
    
    # 创建颜色映射（从白色到深蓝色）
    cm_display = cm.astype(float)
    cm_display = cm_display / cm_display.max() * 100  # 归一化到0-100用于颜色
    
    im = ax.imshow(cm_display, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label='Relative Frequency (%)')
    
    # 设置刻度
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在格子中添加数字
    for i in range(n_classes):
        for j in range(n_classes):
            count = cm[i, j]
            if count > 0:
                percentage = count / cm[i].sum() * 100
                if cm[i, i] == count:  # 对角线（正确分类）
                    text = f'{count:,}\n({percentage:.1f}%)'
                else:  # 非对角线（错误分类）
                    text = f'{count:,}\n({percentage:.1f}%)'
                # 根据背景色自动调整文字颜色
                color = "white" if cm_display[i, j] > 50 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    
    # 计算总体准确率
    overall_acc = np.trace(cm) / total_samples * 100
    
    ax.set_xlabel("Predicted Label", fontsize=18)
    ax.set_ylabel("True Label", fontsize=18)
    # ax.set_title(f"Confusion Matrix - Total Dataset\nTotal Samples: {total_samples:,}, Overall Accuracy: {overall_acc:.2f}%", 
                # fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()

def print_summary(train_loss, train_acc, cm, true_counts, class_names):
    """打印训练摘要"""
    total_samples = sum(true_counts)
    
    print("=" * 80)
    print("RESNET50 TRAINING SUMMARY (50 EPOCHS)")
    print("=" * 80)
    
    print(f"\n📊 DATASET INFORMATION:")
    print(f"  • Total samples: {total_samples:,}")
    print(f"  • Number of classes: 5")
    
    print(f"\n📈 CLASS DISTRIBUTION:")
    for i, (class_name, count) in enumerate(zip(class_names, true_counts)):
        percentage = count / total_samples * 100
        print(f"  • {class_name}: {count:,} samples ({percentage:.1f}%)")
    
    print(f"\n🎯 FINAL RESULTS (Epoch 50):")
    print(f"  • Training Loss: {train_loss[49]:.4f}")
    print(f"  • Training Accuracy: {train_acc[49]:.2f}%")
    
    overall_accuracy = np.trace(cm) / np.sum(cm) * 100
    print(f"  • Overall Dataset Accuracy: {overall_accuracy:.2f}%")
    
    print(f"\n🔍 CONFUSION MATRIX (Total Dataset - {total_samples:,} samples):")
    print("-" * 85)
    print("            Predicted Class")
    print("           ", "  ".join([f"   {i+1}    " for i in range(5)]))
    print("-" * 85)
    
    for i in range(5):
        row = f"True Class {i+1}   "
        for j in range(5):
            row += f"{cm[i, j]:7,d} "
        print(row)
    
    print("-" * 85)
    
    print(f"\n📊 CLASS-WISE ACCURACY:")
    for i in range(5):
        class_acc = cm[i, i] / np.sum(cm[i]) * 100
        print(f"  • {class_names[i]}: {class_acc:.2f}% ({cm[i, i]:,}/{np.sum(cm[i]):,})")
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  • Correct predictions: {np.trace(cm):,}")
    print(f"  • Incorrect predictions: {total_samples - np.trace(cm):,}")
    print(f"  • Overall accuracy: {overall_accuracy:.2f}%")
    
    print("\n" + "=" * 80)

def save_data(train_loss, train_acc, cm, true_counts, filename='training_data.json'):
    """保存训练数据到JSON文件"""
    data = {
        'epochs': list(range(1, 51)),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'confusion_matrix': cm.tolist(),
        'class_distribution': true_counts.tolist(),
        'class_names': ['cft', 'qiu', 'ty', 'yz', 'zft'],
        'total_samples': int(np.sum(true_counts)),
        'final_accuracy': float(np.trace(cm) / np.sum(cm) * 100)
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n💾 Training data saved to {filename}")

def main():
    # 生成训练数据
    epochs = list(range(1, 51))
    train_loss, train_acc = generate_training_data(50)
    
    # 生成混淆矩阵（总样本60800）
    class_names = ['cft', 'qiu', 'ty', 'yz', 'zft']
    cm, true_counts = generate_confusion_matrix(
        n_samples=60800, 
        accuracy=0.9782, 
        n_classes=5
    )
    
    # 打印摘要
    print_summary(train_loss, train_acc, cm, true_counts, class_names)
    
    # 绘制训练曲线
    plot_training_curve(epochs, train_loss, train_acc)
    
    # 绘制混淆矩阵（总数据集）
    plot_confusion_matrix(cm, class_names, np.sum(cm))
    
    # 保存数据
    save_data(train_loss, train_acc, cm, true_counts)
    
    # 打印最后10个epoch的详细数据
    print("\n📋 DETAILED LAST 10 EPOCHS:")
    print("-" * 50)
    print("Epoch  Train Loss  Train Acc(%)")
    print("-" * 50)
    for i in range(40, 50):
        print(f"{i+1:5d}  {train_loss[i]:.4f}      {train_acc[i]:.2f}")
    print("-" * 50)

if __name__ == "__main__":
    main()