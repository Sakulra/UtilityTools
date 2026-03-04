import numpy as np
import pandas as pd
import os
import gc
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class SpectrogramConverter:
    """
    将一维序列转换为时频图像的类
    """
    def __init__(self, img_size=(224, 224), sample_rate=1000, overlap_ratio=0.75):
        """
        参数:
        - img_size: 输出图像尺寸 (高度, 宽度)
        - sample_rate: 采样率 (Hz)，用于频率计算
        - overlap_ratio: 短时傅里叶变换的重叠比例
        """
        self.img_size = img_size
        self.sample_rate = sample_rate
        self.overlap_ratio = overlap_ratio
        
    def compute_spectrogram(self, signal_data, method='stft'):
        """
        计算信号的时频表示
        
        参数:
        - signal_data: 一维信号数据
        - method: 'stft'（短时傅里叶变换）或 'cwt'（连续小波变换）
        
        返回:
        - spectrogram: 时频图像矩阵
        """
        if method == 'stft':
            return self._compute_stft(signal_data)
        elif method == 'cwt':
            return self._compute_cwt(signal_data)
        else:
            raise ValueError(f"不支持的时频分析方法: {method}")
    
    def _compute_stft(self, signal_data):
        """计算STFT频谱图"""
        # 设置STFT参数
        nperseg = min(256, len(signal_data) // 10)  # 每个段的长度
        noverlap = int(nperseg * self.overlap_ratio)  # 重叠长度
        
        # 计算STFT
        f, t, Zxx = signal.stft(
            signal_data,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann'
        )
        
        # 计算幅度谱
        spectrogram = np.abs(Zxx)
        
        # 转换为分贝尺度（更符合人眼感知）
        spectrogram = 20 * np.log10(spectrogram + 1e-10)
        
        return spectrogram
    
    def _compute_cwt(self, signal_data, scales=np.arange(1, 128)):
        """计算连续小波变换"""
        # 使用Ricker小波（墨西哥帽小波）
        widths = scales
        cwtmatr = signal.cwt(signal_data, signal.ricker, widths)
        
        # 取绝对值并转换为分贝尺度
        spectrogram = np.abs(cwtmatr)
        spectrogram = 20 * np.log10(spectrogram + 1e-10)
        
        return spectrogram
    
    def normalize_spectrogram(self, spectrogram):
        """标准化频谱图"""
        # 移除NaN和无穷值
        spectrogram = np.nan_to_num(spectrogram)
        
        # 归一化到[0, 1]范围
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-10)
        
        return spectrogram
    
    def resize_spectrogram(self, spectrogram):
        """调整频谱图大小"""
        from scipy import ndimage
        
        # 使用插值调整大小
        resized = ndimage.zoom(
            spectrogram,
            (
                self.img_size[0] / spectrogram.shape[0],
                self.img_size[1] / spectrogram.shape[1]
            ),
            order=1  # 双线性插值
        )
        
        # 确保输出尺寸正确
        if resized.shape != self.img_size:
            resized = resized[:self.img_size[0], :self.img_size[1]]
            if resized.shape != self.img_size:
                # 如果需要填充
                pad_h = max(0, self.img_size[0] - resized.shape[0])
                pad_w = max(0, self.img_size[1] - resized.shape[1])
                resized = np.pad(
                    resized,
                    ((0, pad_h), (0, pad_w)),
                    mode='constant',
                    constant_values=0
                )
        
        return resized
    
    def convert_to_image(self, signal_data, method='stft'):
        """完整的转换流程：信号 → 频谱图 → 图像"""
        # 1. 计算频谱图
        spectrogram = self.compute_spectrogram(signal_data, method)
        
        # 2. 标准化
        spectrogram = self.normalize_spectrogram(spectrogram)
        
        # 3. 调整大小
        image = self.resize_spectrogram(spectrogram)
        
        # 4. 转换为3通道（模仿RGB图像）
        image_3channel = np.stack([image] * 3, axis=-1)
        
        return image_3channel
    
    def visualize_conversion(self, signal_data, title="Signal and Spectrogram"):
        """可视化转换过程"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 原始信号
        axes[0, 0].plot(signal_data)
        axes[0, 0].set_title("Original Signal")
        axes[0, 0].set_xlabel("Sample")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(True)
        
        # STFT频谱图
        stft_spec = self._compute_stft(signal_data)
        im1 = axes[0, 1].imshow(stft_spec, aspect='auto', cmap='viridis', 
                               extent=[0, len(signal_data)/self.sample_rate, 0, self.sample_rate/2])
        axes[0, 1].set_title("STFT Spectrogram")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Frequency (Hz)")
        plt.colorbar(im1, ax=axes[0, 1])
        
        # CWT频谱图
        cwt_spec = self._compute_cwt(signal_data)
        im2 = axes[1, 0].imshow(cwt_spec, aspect='auto', cmap='hot', 
                               extent=[0, len(signal_data)/self.sample_rate, 1, 128])
        axes[1, 0].set_title("CWT Scalogram")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Scale")
        plt.colorbar(im2, ax=axes[1, 0])
        
        # 最终图像
        final_image = self.convert_to_image(signal_data, 'stft')
        axes[1, 1].imshow(final_image[:, :, 0], cmap='gray')
        axes[1, 1].set_title(f"Final Image ({self.img_size[0]}x{self.img_size[1]})")
        axes[1, 1].set_xlabel("Width")
        axes[1, 1].set_ylabel("Height")
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

def process_csv_files(csv_files, output_dir='spectrogram_data', 
                     img_size=(224, 224), max_samples_per_class=None):
    """
    处理多个CSV文件，将一维序列转换为时频图像
    
    参数:
    - csv_files: CSV文件路径列表
    - output_dir: 输出目录
    - img_size: 输出图像尺寸
    - max_samples_per_class: 每个类别的最大样本数（用于测试）
    """
    import time
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    converter = SpectrogramConverter(img_size=img_size)
    
    all_images = []
    all_labels = []
    all_filenames = []
    
    print("开始处理CSV文件...")
    
    for class_idx, csv_file in enumerate(tqdm(csv_files, desc="处理文件")):
        if not os.path.exists(csv_file):
            print(f"警告: 文件 {csv_file} 不存在，跳过")
            continue
        
        print(f"处理文件: {csv_file} (类别: {class_idx})")
        
        # 分块读取大文件
        chunk_size = 5000
        samples_processed = 0
        
        for chunk in pd.read_csv(csv_file, header=None, chunksize=chunk_size):
            # 提取数据（最后一列是标签，但这里我们根据文件名知道类别）
            data_chunk = chunk.values.astype(np.float32)
            
            for row_idx in range(len(data_chunk)):
                # 限制每个类别的样本数（用于测试）
                if max_samples_per_class and samples_processed >= max_samples_per_class:
                    break
                
                # 获取一维序列（去掉最后一列标签）
                signal_data = data_chunk[row_idx, :-1]
                
                # 转换为时频图像
                try:
                    image = converter.convert_to_image(signal_data, method='stft')
                    all_images.append(image)
                    all_labels.append(class_idx)
                    
                    # 生成文件名
                    filename = f"class_{class_idx}_sample_{samples_processed:06d}.npy"
                    all_filenames.append(filename)
                    
                    samples_processed += 1
                    
                except Exception as e:
                    print(f"处理样本时出错: {e}")
                    continue
            
            # 释放内存
            del chunk
            gc.collect()
            
            # 达到最大样本数限制
            if max_samples_per_class and samples_processed >= max_samples_per_class:
                break
        
        print(f"类别 {class_idx} 处理完成: {samples_processed} 个样本")
    
    if not all_images:
        print("错误: 没有处理任何样本")
        return None
    
    # 转换为数组
    all_images = np.array(all_images, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)
    
    print(f"总样本数: {all_images.shape[0]}")
    print(f"图像形状: {all_images.shape[1:]}")
    
    # 打乱数据
    print("打乱数据...")
    indices = np.random.permutation(len(all_images))
    all_images = all_images[indices]
    all_labels = all_labels[indices]
    all_filenames = [all_filenames[i] for i in indices]
    
    # 划分数据集: 70%训练集, 15%验证集, 15%测试集
    print("划分数据集...")
    n_total = len(all_images)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    train_images = all_images[:n_train]
    train_labels = all_labels[:n_train]
    train_filenames = all_filenames[:n_train]
    
    val_images = all_images[n_train:n_train+n_val]
    val_labels = all_labels[n_train:n_train+n_val]
    val_filenames = all_filenames[n_train:n_train+n_val]
    
    test_images = all_images[n_train+n_val:]
    test_labels = all_labels[n_train+n_val:]
    test_filenames = all_filenames[n_train+n_val:]
    
    print(f"训练集: {len(train_images)} 个样本")
    print(f"验证集: {len(val_images)} 个样本")
    print(f"测试集: {len(test_images)} 个样本")
    
    # 保存数据
    print("保存数据...")
    
    # 保存为NPZ文件（快速加载）
    np.savez(
        os.path.join(output_dir, 'train_data.npz'),
        images=train_images,
        labels=train_labels,
        filenames=train_filenames
    )
    
    np.savez(
        os.path.join(output_dir, 'val_data.npz'),
        images=val_images,
        labels=val_labels,
        filenames=val_filenames
    )
    
    np.savez(
        os.path.join(output_dir, 'test_data.npz'),
        images=test_images,
        labels=test_labels,
        filenames=test_filenames
    )
    
    # 保存为单个图像文件（可选，用于可视化）
    save_images = True
    if save_images:
        for split_name, images, labels, filenames in [
            ('train', train_images, train_labels, train_filenames),
            ('val', val_images, val_labels, val_filenames),
            ('test', test_images, test_labels, test_filenames)
        ]:
            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            for img, label, filename in tqdm(zip(images, labels, filenames), 
                                           desc=f"保存{split_name}图像", 
                                           total=len(images)):
                # 保存为PNG
                img_path = os.path.join(split_dir, filename.replace('.npy', '.png'))
                plt.imsave(img_path, img[:, :, 0], cmap='gray')
    
    # 保存数据信息
    info = {
        'n_classes': len(csv_files),
        'img_size': img_size,
        'n_train': len(train_images),
        'n_val': len(val_images),
        'n_test': len(test_images),
        'class_names': [f'Class_{i}' for i in range(len(csv_files))]
    }
    
    import json
    with open(os.path.join(output_dir, 'data_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    # 可视化几个样本
    print("\n可视化样本...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i in range(6):
        ax = axes[i//3, i%3]
        idx = np.random.randint(0, len(train_images))
        ax.imshow(train_images[idx, :, :, 0], cmap='viridis')
        ax.set_title(f"Class {train_labels[idx]}")
        ax.axis('off')
    plt.suptitle("随机训练样本示例")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_visualization.png'), dpi=150)
    plt.show()
    
    print(f"\n所有处理完成！数据已保存到 {output_dir} 目录")
    
    return info

def main():
    # 设置CSV文件路径
    csv_files = [
        'object1.csv',  # 第一种物体
        'object2.csv',  # 第二种物体
        'object3.csv',  # 第三种物体
        'object4.csv',  # 第四种物体
        'object5.csv'   # 第五种物体
    ]
    
    # 检查文件是否存在
    existing_files = []
    for file in csv_files:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            print(f"警告: 文件 {file} 不存在")
    
    if len(existing_files) == 0:
        print("错误: 没有找到任何CSV文件！")
        return
    
    print(f"找到 {len(existing_files)} 个CSV文件")
    
    # 处理文件
    start_time = time.time()
    info = process_csv_files(
        existing_files,
        output_dir='spectrogram_data',
        img_size=(224, 224),
        max_samples_per_class=2000  # 每个类别最多处理2000个样本（可根据需要调整）
    )
    
    if info:
        end_time = time.time()
        print(f"\n总处理时间: {end_time - start_time:.2f} 秒")
        print(f"\n数据信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()