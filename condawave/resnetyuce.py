import torch
import numpy as np
import os
import json
from train_resnet import SpectrogramDataset, ResNetClassifier, EfficientNetClassifier
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_and_test(model_path, data_dir='spectrogram_data', batch_size=32):
    """加载模型并在测试集上评估"""
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=device)
    info = checkpoint['info']
    
    print(f"模型信息:")
    print(f"类别数: {info['n_classes']}")
    print(f"图像尺寸: {info['img_size']}")
    
    # 确定模型类型
    model_name = 'resnet50'  # 默认
    if 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
    elif 'resnet' in model_path.lower():
        model_name = 'resnet50'
    elif 'efficientnet' in model_path.lower():
        model_name = 'efficientnet_b0'
    
    # 创建模型
    if model_name.startswith('efficientnet'):
        model = EfficientNetClassifier(
            num_classes=info['n_classes'],
            model_name=model_name,
            pretrained=False
        )
    else:
        model = ResNetClassifier(
            num_classes=info['n_classes'],
            backbone=model_name,
            pretrained=False
        )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\n模型加载成功: {model_name}")
    
    # 创建测试数据加载器
    from torchvision import transforms
    
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = SpectrogramDataset(data_dir, val_transform, 'test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"测试集: {len(test_dataset)} 个样本")
    
    # 进行预测
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            
            # 前向传播
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            
            # 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.numpy())
            
            if batch_idx % 10 == 0:
                print(f"处理批次: {batch_idx}/{len(test_loader)}")
    
    # 计算准确率
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets)) * 100
    
    print(f"\n📊 测试结果:")
    print(f"总体准确率: {accuracy:.2f}%")
    
    # 准备类别名称
    class_names = info.get('class_names', [f'Class_{i}' for i in range(info['n_classes'])])
    
    # 打印分类报告
    print("\n📋 详细分类报告:")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('test_confusion_matrix.png', dpi=150)
    plt.show()
    
    # 每个类别的准确率
    print("\n📈 各个类别准确率:")
    for i, class_name in enumerate(class_names):
        class_indices = np.where(np.array(all_targets) == i)[0]
        if len(class_indices) > 0:
            class_correct = sum(np.array(all_preds)[class_indices] == i)
            class_accuracy = 100. * class_correct / len(class_indices)
            print(f"  {class_name}: {class_accuracy:.2f}% ({class_correct}/{len(class_indices)})")
    
    # 保存测试结果
    results = {
        'model_path': model_path,
        'test_accuracy': float(accuracy),
        'class_accuracies': {},
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': all_targets
    }
    
    for i, class_name in enumerate(class_names):
        class_indices = np.where(np.array(all_targets) == i)[0]
        if len(class_indices) > 0:
            class_correct = sum(np.array(all_preds)[class_indices] == i)
            results['class_accuracies'][class_name] = {
                'accuracy': float(100. * class_correct / len(class_indices)),
                'correct': int(class_correct),
                'total': int(len(class_indices))
            }
    
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 测试结果已保存到: test_results.json")
    
    return accuracy, all_preds, all_probs, all_targets, class_names

def predict_single_sequence(model, converter, sequence, device):
    """预测单个序列"""
    # 转换为图像
    image = converter.convert_to_image(sequence, method='stft')
    
    # 转换为tensor
    image_tensor = torch.FloatTensor(image.transpose(2, 0, 1)).unsqueeze(0)
    
    # 标准化
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_tensor)
    
    # 预测
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        prob, pred = torch.max(probs, 1)
    
    return pred.item(), prob.item()

def main():
    # 模型路径
    model_path = 'final_resnet50.pth'  # 或 'best_resnet50.pth'
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        print("请先运行 train_resnet.py 训练模型")
        return
    
    # 测试模型
    accuracy, preds, probs, targets, class_names = load_model_and_test(
        model_path,
        data_dir='spectrogram_data',
        batch_size=32
    )
    
    print(f"\n✅ 测试完成！最终准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    main()