import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from tqdm import tqdm
import argparse

def analyze_signal_parameters(signal_data, default_fs=500000):
    """
    分析信号并建议合适的参数
    """
    # 估计信号的主频（通过FFT）
    n = len(signal_data)
    fft_vals = np.abs(np.fft.fft(signal_data))
    fft_freq = np.fft.fftfreq(n, d=1/default_fs)
    
    # 只考虑正频率
    positive_freq = fft_freq[:n//2]
    positive_fft = fft_vals[:n//2]
    
    # 找到能量最大的频率（主频）
    main_freq = positive_freq[np.argmax(positive_fft)]
    
    return main_freq

def csv_to_stft_images(csv_path, output_dir, img_size=(224, 224), 
                       fs=None, auto_detect=True):
    """
    将CSV文件中的声波数据转换为STFT图像
    
    Parameters:
    csv_path: CSV文件路径
    output_dir: 输出图片目录
    img_size: 输出图片大小
    fs: 采样频率(Hz)，如果为None则自动检测或使用推荐值
    auto_detect: 是否自动检测最佳参数
    """
    
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    for dir_path in [train_dir, val_dir]:
        for i in range(1, 6):
            os.makedirs(os.path.join(dir_path, str(i)), exist_ok=True)
    
    # 读取CSV文件
    print("正在读取CSV文件...")
    df = pd.read_csv(csv_path)
    print(f"数据集形状: {df.shape}")
    
    # 分离特征和标签
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)
    
    print(f"声波数据形状: {X.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 如果自动检测参数
    if auto_detect or fs is None:
        # 取第一个信号分析
        sample_signal = X[0]
        
        # 根据信号长度估计时间分辨率
        signal_length = len(sample_signal)
        time_duration = signal_length / 500000  # 假设初始fs=500kHz估计时长
        
        # 分析信号特性
        from scipy import signal as scipy_signal
        
        # 尝试多种可能的fs值
        possible_fs = [50000, 100000, 250000, 500000, 1000000]
        best_fs = None
        best_spectrum = None
        best_score = 0
        
        print("\n正在分析信号特性以确定最佳采样频率...")
        
        for test_fs in possible_fs:
            # 计算频谱
            f, t, Zxx = scipy_signal.stft(sample_signal, fs=test_fs, 
                                          nperseg=min(256, len(sample_signal)//4),
                                          noverlap=None)
            
            # 评估频谱质量
            magnitude = np.abs(Zxx)
            # 计算频谱的"丰富度"（非零系数的比例）
            non_zero_ratio = np.sum(magnitude > magnitude.mean()) / magnitude.size
            
            # 计算频率覆盖范围
            freq_coverage = np.sum(np.mean(magnitude, axis=1) > magnitude.mean()) / len(f)
            
            # 综合评分
            score = non_zero_ratio * 0.5 + freq_coverage * 0.5
            print(f"  fs={test_fs/1000:.0f}kHz: 评分={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_fs = test_fs
        
        fs = best_fs
        print(f"\n选择的最佳采样频率: {fs/1000:.0f} kHz")
        
        # 根据信号长度动态调整STFT参数
        signal_duration = signal_length / fs
        print(f"信号时长: {signal_duration*1000:.2f} ms")
        
        # 根据信号时长选择适当的窗口大小
        if signal_duration < 0.001:  # <1ms
            nperseg = min(64, signal_length//8)
        elif signal_duration < 0.01:  # 1-10ms
            nperseg = min(128, signal_length//8)
        elif signal_duration < 0.1:  # 10-100ms
            nperseg = min(256, signal_length//8)
        else:  # >100ms
            nperseg = min(512, signal_length//8)
        
        # 确保nperseg是2的幂次
        nperseg = 2 ** int(np.log2(nperseg))
        noverlap = nperseg // 2  # 50%重叠
        
        print(f"STFT参数 - 窗口大小: {nperseg}, 重叠: {noverlap}")
    else:
        # 使用用户指定的参数
        nperseg = min(256, X.shape[1]//8)
        nperseg = 2 ** int(np.log2(nperseg))
        noverlap = nperseg // 2
    
    # 划分训练集和验证集
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # 处理训练集
    print("\n正在生成训练集STFT图像...")
    for idx in tqdm(train_indices):
        signal_data = X[idx]
        label = y[idx]
        
        # 计算STFT
        f, t, Zxx = signal.stft(signal_data, fs=fs, nperseg=nperseg, 
                                noverlap=noverlap, boundary=None)
        
        # 计算幅度谱 (dB)
        magnitude = np.abs(Zxx)
        # 避免log(0)并压缩动态范围
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # 归一化到0-1范围
        magnitude_db_norm = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min() + 1e-10)
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(img_size[0]/100, img_size[1]/100), dpi=100)
        
        # 绘制频谱图（只显示有意义的频率范围，通常0-200kHz）
        f_max_display = min(200000, f.max())  # 限制显示频率范围
        f_idx = f <= f_max_display
        
        im = ax.imshow(magnitude_db_norm[f_idx], aspect='auto', origin='lower',
                      cmap='viridis', 
                      extent=[t.min()*1000, t.max()*1000, f[f_idx].min()/1000, f[f_idx].max()/1000])
        
        # 移除坐标轴和边框
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # 保存图像
        img_path = os.path.join(train_dir, str(label), f'sample_{idx}.png')
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
    
    # 处理验证集
    print("\n正在生成验证集STFT图像...")
    for idx in tqdm(val_indices):
        signal_data = X[idx]
        label = y[idx]
        
        # 计算STFT
        f, t, Zxx = signal.stft(signal_data, fs=fs, nperseg=nperseg, 
                                noverlap=noverlap, boundary=None)
        
        # 计算幅度谱 (dB)
        magnitude = np.abs(Zxx)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # 归一化
        magnitude_db_norm = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min() + 1e-10)
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(img_size[0]/100, img_size[1]/100), dpi=100)
        f_max_display = min(200000, f.max())
        f_idx = f <= f_max_display
        
        ax.imshow(magnitude_db_norm[f_idx], aspect='auto', origin='lower',
                 cmap='viridis',
                 extent=[t.min()*1000, t.max()*1000, f[f_idx].min()/1000, f[f_idx].max()/1000])
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # 保存图像
        img_path = os.path.join(val_dir, str(label), f'sample_{idx}.png')
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
    
    # 保存参数信息
    param_info = {
        'fs': fs,
        'nperseg': nperseg,
        'noverlap': noverlap,
        'signal_length': X.shape[1],
        'num_samples': len(X)
    }
    
    import json
    with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
        json.dump(param_info, f, indent=2)
    
    print(f"\n转换完成!")
    print(f"采样频率: {fs/1000:.0f} kHz")
    print(f"训练集: {len(train_indices)} 张图片保存在 {train_dir}")
    print(f"验证集: {len(val_indices)} 张图片保存在 {val_dir}")
    print(f"参数信息已保存到 {os.path.join(output_dir, 'parameters.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将声波CSV数据转换为STFT图像')
    parser.add_argument('--csv_path', type=str, required=True, help='CSV文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出图片目录')
    parser.add_argument('--fs', type=int, default=None, help='采样频率(Hz)，不指定则自动检测')
    parser.add_argument('--no_auto', action='store_true', help='禁用自动参数检测')
    
    args = parser.parse_args()
    
    csv_to_stft_images(args.csv_path, args.output_dir, 
                       fs=args.fs, 
                       auto_detect=not args.no_auto)