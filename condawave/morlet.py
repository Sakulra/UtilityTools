import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib

# Set font to Times New Roman and font size to 18
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

def morlet_wavelet_transform(data, fs, frequencies):
    """
    Perform Morlet wavelet transform
    """
    dt = 1.0 / fs
    wavelet_scale = 6.0  # Standard Morlet wavelet parameter
    
    # Initialize complex wavelet coefficients matrix
    coefs = np.zeros((len(frequencies), len(data)), dtype=complex)
    
    for i, freq in enumerate(frequencies):
        # Calculate scale from frequency
        scale = wavelet_scale / (2 * np.pi * freq * dt)
        
        # Generate Morlet wavelet and convolve with signal
        wavelet_length = int(scale * 10)  # Wavelet length based on scale
        if wavelet_length % 2 == 0:
            wavelet_length += 1
        
        # Create Morlet wavelet
        t = np.arange(-(wavelet_length-1)//2, (wavelet_length+1)//2) * dt
        wavelet = (np.exp(2j * np.pi * freq * t) * 
                  np.exp(-t**2 / (2 * scale**2 * dt**2)))
        
        # Normalize wavelet
        wavelet = wavelet / np.sqrt(scale * dt)
        
        # Convolve with signal
        convolution = np.convolve(data, np.conj(wavelet), mode='same')
        coefs[i, :] = convolution
    
    return coefs

# Read the first row from CSV file
csv_file = 'E:/shiyan_data/cft_processed_4000.csv'  # Replace with your CSV filename
df = pd.read_csv(csv_file, header=None)
signal_data = df.iloc[0, :].values.astype(float)

# Read time axis from ts4000.txt file
time_axis = np.loadtxt('ts4000.txt')

# Calculate sampling frequency
if len(time_axis) > 1:
    fs = 1.0 / np.mean(np.diff(time_axis))
else:
    fs = 1.0
    time_axis = np.arange(len(signal_data))

# Define frequency range for wavelet transform
fmin = 1  # Minimum frequency in Hz
fmax = fs / 2  # Maximum frequency (Nyquist frequency)
num_freqs = 100  # Number of frequency bins

# Generate logarithmically spaced frequencies
frequencies = np.logspace(np.log10(fmin), np.log10(fmax), num_freqs)

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
axes[0].set_xlabel('Time [s]', fontsize=18)
axes[0].set_ylabel('Amplitude', fontsize=18)
axes[0].set_title('Original Signal (First Row of CSV)', fontsize=20, fontweight='bold')
axes[0].grid(True, alpha=0.3, linestyle='--')

# Plot 2: Morlet wavelet transform (magnitude)
pcm1 = axes[1].pcolormesh(time_axis, frequencies, wavelet_magnitude, 
                          cmap='jet', shading='gouraud')
cbar1 = plt.colorbar(pcm1, ax=axes[1])
cbar1.set_label('Magnitude', fontsize=16)
axes[1].set_xlabel('Time [s]', fontsize=18)
axes[1].set_ylabel('Frequency [Hz]', fontsize=18)
axes[1].set_title('Morlet Wavelet Transform - Magnitude', fontsize=20, fontweight='bold')
axes[1].set_yscale('log')  # Logarithmic frequency scale
axes[1].grid(True, alpha=0.3, linestyle='--')

# Plot 3: Morlet wavelet transform (power in dB)
pcm2 = axes[2].pcolormesh(time_axis, frequencies, wavelet_power_db, 
                          cmap='viridis', shading='gouraud', vmin=-20, vmax=20)
cbar2 = plt.colorbar(pcm2, ax=axes[2])
cbar2.set_label('Power [dB]', fontsize=16)
axes[2].set_xlabel('Time [s]', fontsize=18)
axes[2].set_ylabel('Frequency [Hz]', fontsize=18)
axes[2].set_title('Morlet Wavelet Transform - Power (dB)', fontsize=20, fontweight='bold')
axes[2].set_yscale('log')  # Logarithmic frequency scale
axes[2].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('morlet_wavelet_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a more detailed single plot for wavelet transform
fig2, ax = plt.subplots(figsize=(14, 8))

pcm = ax.pcolormesh(time_axis, frequencies, wavelet_power_db, 
                    cmap='jet', shading='gouraud', vmin=-20, vmax=20)
cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label('Power [dB]', fontsize=18)

ax.set_xlabel('Time [s]', fontsize=18)
ax.set_ylabel('Frequency [Hz]', fontsize=18)
ax.set_title('Morlet Wavelet Transform', fontsize=20, fontweight='bold')
ax.set_yscale('log')  # Logarithmic frequency scale for better visualization
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('morlet_wavelet_main.png', dpi=300, bbox_inches='tight')
plt.show()

# Print analysis information
# print("=" * 60)
# print("MORLET WAVELET TRANSFORM ANALYSIS RESULTS")
# print("=" * 60)
# print(f"Signal length: {len(signal_data)} samples")
# print(f"Time duration: {time_axis[-1] - time_axis[0]:.2f} s")
# print(f"Sampling frequency: {fs:.2f} Hz")
# print(f"Number of frequency bins: {len(frequencies)}")
# print(f"Frequency range: {frequencies[0]:.2f} - {frequencies[-1]:.2f} Hz")
# print(f"Time points: {len(time_axis)}")
# print(f"Wavelet coefficient matrix shape: {wavelet_coefs.shape}")
# print("=" * 60)

# Save results
# np.save('frequencies.npy', frequencies)
# np.save('time_axis.npy', time_axis)
# np.save('wavelet_coefs.npy', wavelet_coefs)
# np.save('wavelet_power.npy', wavelet_power)

# print("\nResults saved to:")
# print("- frequencies.npy")
# print("- time_axis.npy")
# print("- wavelet_coefs.npy")
# print("- wavelet_power.npy")