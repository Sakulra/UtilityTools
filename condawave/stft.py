import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib

# Set font to Times New Roman and font size to 18
matplotlib.rcParams['font.family'] = ['Times New Roman',"SimSun"]
matplotlib.rcParams['font.size'] = 24

# Read the first row from CSV file
csv_data = pd.read_csv('D:/shiyan_data/剔除多余数据/cft_processed_4000.csv', header=None)
first_row = csv_data.iloc[4, :].values  # Get first row data

# Read time axis from data.txt file
time_data = np.loadtxt('ts4000.txt')

# Perform STFT transformation
frequencies, times, stft_matrix = signal.stft(
    first_row, 
    # fs=1/(time_data[1]-time_data[0]) if len(time_data)>1 else 1.0,  # Sampling frequency
    fs=200000,       # 采样频率 FS = 200000
    nperseg=256,     # 窗口长度 NFFT = 256
    noverlap=192     # 重叠长度 = NFFT - HOP = 256 - 64 = 192
)

# Calculate magnitude of STFT
stft_magnitude = np.abs(stft_matrix)

# # Plot the STFT result
plt.figure(figsize=(12, 8))
plt.pcolormesh(times, frequencies, 20 * np.log10(stft_magnitude + 1e-10), 
               shading='gouraud', cmap='viridis')
plt.colorbar(label='幅度(dB)')
plt.xlabel('时间(s)')
plt.ylabel('频率[Hz]')
# plt.title('STFT of First Row Data')
plt.tight_layout()
plt.savefig('stft_result.png', dpi=300, bbox_inches='tight')
plt.show()

# plot original signal
plt.figure(figsize=(12, 8))
plt.margins(x=0)
plt.plot(time_data-0.025, first_row, linewidth=2, color='blue')
plt.xlabel('时间(s)')
plt.ylabel('幅值(V)')
# plt.title('First Row Data vs Time', fontsize=18)
plt.grid(True, alpha=0.3)#在图上添加网格线便于看
plt.tick_params(axis='both', which='major', labelsize=24)

plt.tight_layout()
plt.show()


# Optional: Save STFT results
# np.save('frequencies.npy', frequencies)
# np.save('times.npy', times)
# np.save('stft_magnitude.npy', stft_magnitude)