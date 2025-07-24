# import pandas as pd

# data = pd.read_csv('E:\shiyan_data\processA/add_labeled_dataA.csv')
# print(data.shape)

import numpy as np
import matplotlib.pyplot as plt

# 参数设置
A0 = 1.0      # 振幅
f0 = 10       # 正弦波频率 (Hz)
T = 1.0       # 脉冲宽度 (s)
fs = 1000     # 采样率 (Hz)

# 生成时间轴
t = np.linspace(-0.5, 1.5, int(2 * fs))  # 时间范围：-0.5s 到 1.5s

# 生成信号
sine_wave = A0 * np.sin(2 * np.pi * f0 * t)
rect_window = np.where((t >= 0) & (t <= T), 1, 0)  # 矩形窗
s_t = sine_wave * rect_window

# 绘图
plt.figure(figsize=(10, 4))
plt.plot(t, s_t, 'b-', linewidth=1.5, label='Sine Pulse Signal')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Pulse Start (t=0)')
plt.axvline(T, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Pulse End (t=T)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sine Pulse Signal $s(t) = A_0 \sin(2\pi f_0 t) \cdot \mathrm{rect}(t/T)$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()