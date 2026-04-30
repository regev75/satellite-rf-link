# Tutorial for Generating Constellation, Eye Diagram, PSD, and BER Curve Plots

This tutorial demonstrates how to generate various plots related to satellite RF link analysis using Python libraries such as Matplotlib, NumPy, and SciPy. The plots covered are:

1. **Constellation Plot**
2. **Eye Diagram Plot**
3. **Power Spectral Density (PSD) Plot**
4. **Bit Error Rate (BER) Curve Plot**

## 1. Constellation Plot

A constellation plot is used to visualize the modulation scheme of a communication system. Here's how to create one:

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample parameters
num_symbols = 1000
symbols = np.random.choice([1, -1], num_symbols)  # BPSK Modulation

# Create constellation plot
plt.figure(figsize=(8, 8))
plt.scatter(symbols.real, symbols.imag)
plt.title('Constellation Plot')
plt.xlabel('In-Phase')
plt.ylabel('Quadrature')
plt.grid()
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()
```

## 2. Eye Diagram Plot

An eye diagram provides insights into the signal quality and timing. Below is an example code:

```python
def plot_eye_diagram(signal, sample_rate, pulse_shape='raised_cosine'):
    t = np.arange(0, len(signal) / sample_rate, 1 / sample_rate)
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal)
    plt.title('Eye Diagram')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()
```

## 3. Power Spectral Density (PSD) Plot

The PSD plot shows the power distribution of a signal versus frequency. Here's how to plot it:

```python
from scipy.signal import periodogram

# Generate sample signal
fs = 1000  # Sampling frequency
f = np.linspace(0, fs / 2, 1000)
signal = np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.5, len(t))

# Calculate PSD
frequencies, psd = periodogram(signal, fs)

plt.figure(figsize=(10, 6))
plt.plot(frequencies, 10 * np.log10(psd))  # Convert to dB
plt.title('Power Spectral Density (PSD)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.grid()
plt.show()
```

## 4. Bit Error Rate (BER) Curve Plot

A BER curve plot shows the performance of a communication system over various signal-to-noise ratios (SNR). Here’s an example:

```python
snr_values = np.arange(0, 10, 1)
ber_values = 0.5 * np.exp(-snr_values)  # Hypothetical BER values

plt.figure(figsize=(10, 6))
plt.semilogy(snr_values, ber_values, 'r-')
plt.title('Bit Error Rate (BER) Curve')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.grid()
plt.show()
```

## Conclusion
This tutorial provided step-by-step guidance on generating constellation, eye diagram, PSD, and BER curve plots essential for analyzing satellite RF links. By following these examples, you can create these visualizations for your specific data and parameters.