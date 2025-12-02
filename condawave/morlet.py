import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pywt

# 读取CSV文件的第一行数据
try:
    # 请将'your_file.csv'替换为你的实际文件名
    df = pd.read_csv('E:\shiyan_data\processA\cft_processed_data.csv', nrows=1)
    
    # 提取第一行数据作为信号
    signal_data = df.iloc[0].values
    print(f"Signal length: {len(signal_data)}")
    print(f"Signal data: {signal_data}")
    
    # 创建2x1的子图画布
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 绘制原始信号
    axes[0].plot(signal_data, 'b-', linewidth=1.5, label='Original Signal')
    axes[0].scatter(range(len(signal_data)), signal_data, color='red', s=5, alpha=0.7, label='Data Points')
    axes[0].set_title('Original Signal (Row 1)')
    axes[0].set_xlabel('Data Point Index')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 进行Morlet小波变换
    # 定义频率范围
    frequencies = np.linspace(0.1, 10, 100)  # 根据你的数据调整频率范围
    
    # 使用scipy的cwt函数进行Morlet小波变换
    widths = 5 * frequencies  # 小波宽度与频率相关
    cwtmatr = signal.cwt(signal_data, signal.morlet2, widths)
    
    # 绘制小波变换结果
    im = axes[1].imshow(np.abs(cwtmatr), aspect='auto', cmap='viridis',
                       extent=[0, len(signal_data), frequencies[-1], frequencies[0]])
    axes[1].set_title('Morlet Wavelet Transform')
    axes[1].set_xlabel('Time/Sample Points')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].grid(False)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('Magnitude')
    
    # 调整布局
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("文件未找到，请检查文件名和路径是否正确")
except Exception as e:
    print(f"发生错误: {e}")

# 使用pywt库的替代方案（如果需要更专业的小波变换）
def morlet_transform_pywt(signal_data):
    """使用pywt库进行Morlet小波变换"""
    try:
        # 定义小波
        wavelet = 'morl'
        
        # 进行连续小波变换
        scales = np.arange(1, 128)  # 尺度范围
        coefficients, frequencies = pywt.cwt(signal_data, scales, wavelet)
        
        return coefficients, frequencies
    except Exception as e:
        print(f"pywt小波变换错误: {e}")
        return None, None

# 可选：使用pywt库的版本
try:
    # 重新读取数据
    df = pd.read_csv('your_file.csv', nrows=1)
    signal_data = df.iloc[0].values
    
    # 使用pywt进行Morlet小波变换
    coefficients, frequencies = morlet_transform_pywt(signal_data)
    
    if coefficients is not None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 原始信号
        axes[0].plot(signal_data, 'b-', linewidth=1.5)
        axes[0].scatter(range(len(signal_data)), signal_data, color='red', s=5, alpha=0.7)
        axes[0].set_title('Original Signal')
        axes[0].set_xlabel('Data Point Index')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # 小波变换结果
        im = axes[1].imshow(np.abs(coefficients), aspect='auto', cmap='hot',
                           extent=[0, len(signal_data), frequencies[-1], frequencies[0]])
        axes[1].set_title('Morlet CWT Result')
        axes[1].set_xlabel('Time/Samples')
        axes[1].set_ylabel('Frequency')
        plt.colorbar(im, ax=axes[1]).set_label('Mag')
        
        plt.tight_layout()
        plt.show()
        
except Exception as e:
    print(f"pywt版本错误: {e}")