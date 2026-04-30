import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def plot_constellation(data, title='Constellation Diagram'):
    """Plot a constellation diagram.
    
    Parameters:
        data : np.ndarray
            Complex symbols to be plotted.
        title : str
            Title of the plot.
    """    
    plt.figure()
    plt.scatter(data.real, data.imag)
    plt.title(title)
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.grid()
    plt.axis('equal')
    plt.show()  


def plot_eye_diagram(data, samples_per_symbol, title='Eye Diagram'):
    """Plot an eye diagram.
    
    Parameters:
        data : np.ndarray
            Received samples of the signal.
        samples_per_symbol : int
            Number of samples per symbol.
        title : str
            Title of the plot.
    """    
    plt.figure()
    num_symbols = len(data) // samples_per_symbol
    for i in range(num_symbols - 1):
        plt.plot(data[i * samples_per_symbol:(i + 1) * samples_per_symbol])
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()


def plot_psd(data, fs, title='Power Spectral Density'):
    """Plot the Power Spectral Density using Welch's method.
    
    Parameters:
        data : np.ndarray
            Signal data.
        fs : float
            Sampling frequency.
        title : str
            Title of the plot.
    """  
    plt.figure()
    f, Pxx = signal.welch(data, fs, nperseg=1024)
    plt.semilogy(f, Pxx)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V**2/Hz)')
    plt.grid()
    plt.show()  


def plot_ber_curve(snr, ber, title='BER Curve'):
    """Plot a Bit Error Rate (BER) vs SNR curve.
    
    Parameters:
        snr : np.ndarray
            Signal to Noise Ratio values.
        ber : np.ndarray
            Bit Error Rate corresponding to the SNR values.
        title : str
            Title of the plot.
    """  
    plt.figure()
    plt.semilogy(snr, ber, marker='o')
    plt.title(title)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.grid()
    plt.show()  
