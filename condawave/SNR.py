# 画出信噪比曲线
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Times New Roman', "SimSun"]
plt.rcParams['font.size'] = 24

# 读取.mat文件
data = scipy.io.loadmat('D:/shiyan_data/cft/18k2024年7月8日13时21分49.mat')

# 查看文件内容
print(data.keys())

# 提取声波数据和噪声数据
signal = data['receive_A'][5200:45200, 0]      # 信号段
noise  = data['receive_A'][50000:90000, 0]     # 纯噪声段（用于估算噪声水平）
time   = data['ts'][0, 5200:45200]             # 信号对应的时间轴

# ==================== 参数设置 ====================
frame_len = 1024      # 窗口长度（点数）
hop_len   = 512       # 窗口滑动步长（点数）
eps       = 1e-12     # 避免除零

# ==================== 计算噪声电压有效值 (RMS) ====================
noise_rms = np.sqrt(np.mean(noise**2))
print(f"噪声有效值 (RMS): {noise_rms:.6f}")

# ==================== 滑动窗口计算信号 RMS 和 SNR ====================
num_frames = (len(signal) - frame_len) // hop_len + 1
snr_db = np.zeros(num_frames)
frame_time = np.zeros(num_frames)      # 每帧中点对应的时间

for i in range(num_frames):
    start = i * hop_len
    end   = start + frame_len
    frame = signal[start:end]
    
    # 信号有效值
    signal_rms = np.sqrt(np.mean(frame**2))
    
    # 信噪比 (dB) = 20 * log10(信号RMS / 噪声RMS)
    snr_db[i] = 20 * np.log10(signal_rms / noise_rms + eps)
    
    # 计算该帧中点对应的时间
    frame_time[i] = np.mean(time[start:end])

# ==================== 绘制信噪比曲线 ====================
plt.figure(figsize=(12, 5))
plt.plot(frame_time, snr_db, linewidth=1.5, color='darkred')
plt.xlabel('时间 (s)', fontsize=24)
plt.ylabel('信噪比(dB)', fontsize=24)
# plt.title('Signal-to-Noise Ratio Variation over Time', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 输出统计信息
print(f"平均信噪比 (dB): {np.mean(snr_db):.2f} dB")
print(f"最大信噪比 (dB): {np.max(snr_db):.2f} dB")
print(f"最小信噪比 (dB): {np.min(snr_db):.2f} dB")

plt.show()