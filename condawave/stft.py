import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib

# Set font to Times New Roman and font size to 18
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 18

# Read the first row from CSV file
csv_data = pd.read_csv('E:/shiyan_data/cft_processed_4000.csv', header=None)  # Replace with your CSV filename
first_row = csv_data.iloc[0, :].values  # Get first row data

# Read time axis from data.txt file
time_data = np.loadtxt('ts4000.txt')

# Perform STFT transformation
frequencies, times, stft_matrix = signal.stft(
    first_row, 
    fs=1/(time_data[1]-time_data[0]) if len(time_data)>1 else 1.0,  # Sampling frequency
    nperseg=256,  # Window length
    noverlap=128  # Overlap between windows
)

# Calculate magnitude of STFT
stft_magnitude = np.abs(stft_matrix)

# # Plot the STFT result
plt.figure(figsize=(12, 8))
plt.pcolormesh(times, frequencies, 20 * np.log10(stft_magnitude + 1e-10), 
               shading='gouraud', cmap='viridis')
plt.colorbar(label='Magnitude [dB]')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
# plt.title('STFT of First Row Data')
plt.tight_layout()
plt.savefig('stft_result.png', dpi=300, bbox_inches='tight')
plt.show()

#plot original signal
# plt.figure(figsize=(12, 8))
# plt.plot(time_data, first_row, linewidth=2, color='blue')
# plt.xlabel('Time', fontsize=18)
# plt.ylabel('Value', fontsize=18)
# # plt.title('First Row Data vs Time', fontsize=18)
# plt.grid(True, alpha=0.3)#在图上添加网格线便于看
# plt.tick_params(axis='both', which='major', labelsize=18)

# plt.tight_layout()
# plt.show()


# Optional: Save STFT results
# np.save('frequencies.npy', frequencies)
# np.save('times.npy', times)
# np.save('stft_magnitude.npy', stft_magnitude)