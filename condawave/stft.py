import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import librosa

# 读取CSV文件的第一行数据
try:
    # 请将'your_file.csv'替换为你的实际文件名
    df = pd.read_csv('E:\shiyan_data\processA\cft_processed_data.csv', nrows=1)
    
    # 提取第一行数据作为信号
    signal_data = df.iloc[0].values.astype(float)
    print(f"读取到的信号长度: {len(signal_data)}")
    print(f"信号数据范围: [{np.min(signal_data):.4f}, {np.max(signal_data):.4f}]")
    
    # 创建2x1的子图画布
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 绘制原始信号
    axes[0].plot(signal_data, 'b-', linewidth=1.5, label='原始信号')
    axes[0].scatter(range(len(signal_data)), signal_data, color='red', s=30, alpha=0.7, label='离散点')
    axes[0].set_title('原始信号 (第一行数据)')
    axes[0].set_xlabel('数据点索引')
    axes[0].set_ylabel('幅值')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 进行STFT变换
    # 设置STFT参数
    fs = 1.0  # 采样率（假设为1Hz，根据你的数据调整）
    nperseg = min(256, len(signal_data) // 4)  # 窗口长度
    noverlap = nperseg // 2  # 重叠长度
    
    # 执行STFT
    f, t, Zxx = signal.stft(signal_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # 绘制STFT结果（频谱图）
    im = axes[1].pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
    axes[1].set_title('STFT变换结果 (频谱图)')
    axes[1].set_xlabel('时间 [秒]')
    axes[1].set_ylabel('频率 [Hz]')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('幅值')
    
    # 调整布局
    plt.tight_layout()
    plt.show()
    
    # 打印STFT参数信息
    print(f"STFT参数: 窗口长度={nperseg}, 重叠={noverlap}")
    print(f"频率范围: [{f[0]:.2f}, {f[-1]:.2f}] Hz")
    print(f"时间范围: [{t[0]:.2f}, {t[-1]:.2f}] 秒")

except FileNotFoundError:
    print("文件未找到，请检查文件名和路径是否正确")
except Exception as e:
    print(f"发生错误: {e}")

# 使用librosa库的STFT版本（提供更多音频处理功能）
def stft_librosa_version(signal_data, sr=1.0):
    """使用librosa进行STFT变换"""
    try:
        # 执行STFT
        D = librosa.stft(signal_data.astype(float), n_fft=min(2048, len(signal_data)))
        
        # 转换为幅度谱
        magnitude = np.abs(D)
        
        # 获取频率和时间轴
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=min(2048, len(signal_data)))
        times = librosa.times_like(D, sr=sr)
        
        return magnitude, frequencies, times
        
    except Exception as e:
        print(f"librosa STFT错误: {e}")
        return None, None, None

# 可选：使用librosa库的版本
try:
    # 重新读取数据
    df = pd.read_csv('your_file.csv', nrows=1)
    signal_data = df.iloc[0].values.astype(float)
    
    # 使用librosa进行STFT
    magnitude, frequencies, times = stft_librosa_version(signal_data)
    
    if magnitude is not None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 原始信号
        axes[0].plot(signal_data, 'b-', linewidth=1.5)
        axes[0].scatter(range(len(signal_data)), signal_data, color='red', s=30, alpha=0.7)
        axes[0].set_title('原始信号 (librosa版本)')
        axes[0].set_xlabel('数据点索引')
        axes[0].set_ylabel('幅值')
        axes[0].grid(True, alpha=0.3)
        
        # STFT结果（使用对数刻度，更适合音频数据）
        im = axes[1].pcolormesh(times, frequencies, librosa.amplitude_to_db(magnitude, ref=np.max), 
                               shading='gouraud', cmap='magma')
        axes[1].set_title('STFT变换结果 (dB尺度)')
        axes[1].set_xlabel('时间 [秒]')
        axes[1].set_ylabel('频率 [Hz]')
        plt.colorbar(im, ax=axes[1], format='%+2.0f dB')
        
        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"librosa版本错误: {e}")