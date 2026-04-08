import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib

# Set font to Times New Roman and font size to 18
plt.rcParams['font.family'] = ['Times New Roman', "SimSun"]
plt.rcParams['font.size'] = 24
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

def morlet_wavelet_transform(data, fs, frequencies, max_wavelet_length=1000):
    """
    Perform Morlet wavelet transform
    
    Parameters:
    -----------
    data : array
        Input signal
    fs : float
        Sampling frequency
    frequencies : array
        Frequencies to analyze
    max_wavelet_length : int
        Maximum wavelet length to use (for performance)
    """
    dt = 1.0 / fs
    n = len(data)
    
    # Initialize complex wavelet coefficients matrix
    coefs = np.zeros((len(frequencies), n), dtype=complex)
    
    for i, freq in enumerate(frequencies):
        # Calculate scale from frequency
        # Standard Morlet parameter: f0 = 1, so scale = f0/freq = 1/freq
        scale = 1.0 / freq  # In seconds
        
        # Calculate wavelet length (number of cycles)
        # Use a fixed number of cycles for better frequency resolution
        n_cycles = 7  # Number of wavelet cycles (adjustable)
        wavelet_length = int(n_cycles * fs / freq)
        
        # Ensure wavelet length is odd and within limits
        if wavelet_length > max_wavelet_length:
            wavelet_length = max_wavelet_length
        if wavelet_length > n:
            wavelet_length = n
        
        if wavelet_length % 2 == 0:
            wavelet_length += 1
        
        # Create time array for wavelet
        t = np.arange(-(wavelet_length-1)//2, (wavelet_length+1)//2) / fs
        
        # Create Morlet wavelet (more standard implementation)
        sigma = n_cycles / (2 * np.pi * freq)  # Gaussian window width
        wavelet = (np.exp(2j * np.pi * freq * t) * 
                  np.exp(-t**2 / (2 * sigma**2)))
        
        # Normalize wavelet to unit energy
        wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet)**2) * dt)
        
        # Convolve with signal using FFT for better performance
        # Pad both signals to length N + wavelet_length - 1
        n_conv = n + wavelet_length - 1
        data_padded = np.zeros(n_conv, dtype=complex)
        wavelet_padded = np.zeros(n_conv, dtype=complex)
        
        data_padded[:n] = data
        wavelet_padded[:wavelet_length] = wavelet[::-1]  # Reverse for convolution
        
        # FFT convolution
        data_fft = np.fft.fft(data_padded)
        wavelet_fft = np.fft.fft(wavelet_padded)
        convolution = np.fft.ifft(data_fft * wavelet_fft)
        
        # Keep only the valid part (same length as original)
        coefs[i, :] = convolution[wavelet_length//2:wavelet_length//2 + n]
    
    return coefs

# Read the first row from CSV file
csv_file = 'E:/shiyan_data/cft_processed_4000.csv'  # Replace with your CSV filename
df = pd.read_csv(csv_file, header=None)
signal_data = df.iloc[0, :].values.astype(float)

# Read time axis from ts4000.txt file
time_axis = np.loadtxt('ts4000.txt')

# Check signal length
print(f"Signal length: {len(signal_data)}")
print(f"Time axis length: {len(time_axis)}")

# Calculate sampling frequency
if len(time_axis) > 1:
    fs = 1.0 / np.mean(np.diff(time_axis))
else:
    fs = 1.0
    time_axis = np.arange(len(signal_data))

print(f"Sampling frequency: {fs:.2f} Hz")

# Define frequency range for wavelet transform
fmin = 1  # Minimum frequency in Hz
fmax = min(fs / 2, 200)  # Maximum frequency (limit to reasonable value)
num_freqs = 50  # Number of frequency bins

# Generate linearly spaced frequencies (better for low frequencies)
frequencies = np.linspace(fmin, fmax, num_freqs)

# Perform Morlet wavelet transform
print("Performing Morlet wavelet transform...")
wavelet_coefs = morlet_wavelet_transform(signal_data, fs, frequencies)

# Calculate magnitude
wavelet_magnitude = np.abs(wavelet_coefs)
wavelet_power = wavelet_magnitude ** 2
wavelet_power_db = 10 * np.log10(wavelet_power + 1e-10)

# Create plots
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Plot 1: Original signal
axes[0].plot(time_axis, signal_data, 'b-', linewidth=1.5)
axes[0].set_xlabel('时间(s)')
axes[0].set_ylabel('电压(V)')
axes[0].grid(True, alpha=0.3, linestyle='--')

# Plot 2: Morlet wavelet transform (magnitude)
pcm1 = axes[1].pcolormesh(time_axis, frequencies, wavelet_magnitude, 
                          cmap='jet', shading='auto')
cbar1 = plt.colorbar(pcm1, ax=axes[1])
cbar1.set_label('Magnitude', fontsize=16)
axes[1].set_xlabel('时间(s)')
axes[1].set_ylabel('频率(Hz)')
axes[1].grid(True, alpha=0.3, linestyle='--')

# Plot 3: Morlet wavelet transform (power in dB)
pcm2 = axes[2].pcolormesh(time_axis, frequencies, wavelet_power_db, 
                          cmap='viridis', shading='auto', vmin=-20, vmax=20)
cbar2 = plt.colorbar(pcm2, ax=axes[2])
cbar2.set_label('功率(dB)', fontsize=16)
axes[2].set_xlabel('时间(s)')
axes[2].set_ylabel('频率(Hz)')
axes[2].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('morlet_wavelet_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a more detailed single plot for wavelet transform
fig2, ax = plt.subplots(figsize=(14, 8))

pcm = ax.pcolormesh(time_axis, frequencies, wavelet_power_db, 
                    cmap='jet', shading='auto', vmin=-20, vmax=20)
cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label('功率(dB)', fontsize=18)

ax.set_xlabel('时间(s)', fontsize=18)
ax.set_ylabel('频率(Hz)', fontsize=18)
ax.set_title('Morlet小波变换', fontsize=20, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('morlet_wavelet_main.png', dpi=300, bbox_inches='tight')
plt.show()

# Print some statistics
print("\nWavelet transform statistics:")
print(f"Power range: {wavelet_power_db.min():.2f} to {wavelet_power_db.max():.2f} dB")
print(f"Frequency range: {frequencies[0]:.2f} to {frequencies[-1]:.2f} Hz")