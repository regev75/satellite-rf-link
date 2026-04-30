import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Function to generate a constellation plot

def constellation_plot(symbols, title="Constellation Plot"):
    plt.scatter(np.real(symbols), np.imag(symbols), color='blue')
    plt.title(title)
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.grid()
    plt.axis('equal')
    plt.show()

# Function to generate an eye diagram

def eye_diagram(signal, samples_per_symbol, title="Eye Diagram"):
    plt.figure()
    for i in range(0, len(signal) - samples_per_symbol, samples_per_symbol):
        plt.plot(signal[i:i+samples_per_symbol], color='blue', alpha=0.5)
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

# Function to generate a Power Spectral Density plot

def psd_plot(signal, title="Power Spectral Density Plot"):
    plt.figure()
    plt.psd(signal, NFFT=1024, Fs=1, color='blue')
    plt.title(title)
    plt.grid()
    plt.show()

# Function to generate a BER curve

def ber_curve(EbN0_dB, title="BER Curve"):
    ber = 0.5 * np.exp(-10**(EbN0_dB/10))
    plt.semilogy(EbN0_dB, ber, color='blue')
    plt.title(title)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Bit Error Rate')
    plt.grid()
    plt.show()

# Example usage
if __name__ == '__main__':
    # Generate example data for constellation
    num_symbols = 1000
    symbols = np.random.normal(0, 1, num_symbols) + 1j * np.random.normal(0, 1, num_symbols)
    constellation_plot(symbols)
    
    # Generate example data for eye diagram
    samples_per_symbol = 100
    signal = np.sin(2 * np.pi * np.linspace(0, 1, num_symbols))
    eye_diagram(signal, samples_per_symbol)
    
    # Generate example data for PSD
    psd_plot(signal)
    
    # Generate example data for BER curve
    EbN0_dB = np.linspace(0, 10, 10)
    ber_curve(EbN0_dB)