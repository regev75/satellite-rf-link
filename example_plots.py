import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Function to create and display constellation diagram
def plot_constellation(data, title='Constellation Diagram'):
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(data), np.imag(data), color='blue')
    plt.title(title)
    plt.xlabel('In-phase')
    plt.ylabel('Quadrature')
    plt.grid()
    plt.axis('equal')
    plt.show()

# Function to create and display eye diagram
def plot_eye_diagram(signal, title='Eye Diagram', samples_per_symbol=100):
    plt.figure(figsize=(10, 4))
    for i in range(0, len(signal) - samples_per_symbol, samples_per_symbol):
        plt.plot(signal[i:i + samples_per_symbol])
    plt.title(title)
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

# Function to create and display Power Spectral Density (PSD)
def plot_psd(signal, title='Power Spectral Density'):
    plt.figure(figsize=(8, 4))
    f, Pxx = signal.welch(signal, fs=1.0, nperseg=256)
    plt.semilogy(f, Pxx)
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('PSD (V**2/Hz)')
    plt.grid()
    plt.show()

# Function to create and display BER curve
def plot_ber_curve(snr_db, ber, title='BER Curve'):
    plt.figure(figsize=(8, 4))
    plt.semilogy(snr_db, ber, marker='o')
    plt.title(title)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.grid()
    plt.show()

# Example data generation

# Generate random QPSK symbols
num_symbols = 1000
data = (np.random.randint(0, 2, num_symbols) * 2 - 1) + 1j * (np.random.randint(0, 2, num_symbols) * 2 - 1)

# Plotting
plot_constellation(data)

# Eye diagram (example signal)
eye_signal = np.repeat(data, 100)
plot_eye_diagram(eye_signal)

# PSD
plot_psd(eye_signal)

# BER curve example data
snr_db = np.arange(0, 11, 1)
ber = np.exp(-0.1 * snr_db)
plot_ber_curve(snr_db, ber)