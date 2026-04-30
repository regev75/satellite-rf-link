# 📡 Satellite RF Link

A comprehensive Python implementation of an RF satellite link
receiver/transmitter system, modelled on the MATLAB RF Satellite Link example
([MathWorks documentation](https://www.mathworks.com/help/comm/ug/rf-satellite-link.html)).

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                   Satellite Downlink Transmitter                   │
│  random bits → QPSK modulation → RRC pulse shaping → TX waveform  │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                       Propagation Channel                          │
│          Free-space path loss  │  AWGN  │  Doppler shift           │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                 Ground Station Downlink Receiver                   │
│   AGC → carrier offset correction → RRC matched filter → QPSK     │
│   demodulation → BER / SNR / EVM metrics                          │
└────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

| File | Description |
|------|-------------|
| `config.py` | System-wide parameters (orbit presets, frequency bands, defaults) |
| `satellite_transmitter.py` | QPSK modulator, RRC pulse shaping, RF up-conversion |
| `propagation_channel.py` | Free-space path loss, AWGN, Doppler shift |
| `satellite_receiver.py` | AGC, carrier recovery, RRC matched filter, QPSK demod, BER/EVM |
| `signal_analysis.py` | Constellation, eye diagram, PSD, BER curve plots |
| `simulate_link.py` | End-to-end simulation with CLI interface |
| `test_satellite_link.py` | pytest unit & integration tests |
| `requirements.txt` | Python dependencies |

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the simulation

```bash
python simulate_link.py
```

With custom options:

```bash
python simulate_link.py --bits 20000 --snr-min 0 --snr-max 16 \
                        --orbit LEO --band Ka --doppler
```

### Run tests

```bash
pytest test_satellite_link.py -v
```

---

## Module Overview

### `satellite_transmitter.py`

- **`generate_bits(n)`** – generate random binary data
- **`qpsk_modulate(bits)`** – Gray-coded QPSK symbol mapping
- **`pulse_shape(symbols, sps, rolloff, span)`** – RRC upsampling + filtering
- **`modulate_rf(baseband, carrier_freq, sample_rate)`** – complex → real passband
- **`compute_psd(signal, sample_rate)`** – Welch PSD estimate
- **`SatelliteTransmitter`** – convenience class wrapping the full TX chain

### `propagation_channel.py`

- **`free_space_path_loss_db(distance_m, frequency_hz)`** – Friis FSPL
- **`doppler_shift_hz(carrier, velocity, elevation)`** – Doppler prediction
- **`PropagationChannel`** – full channel model with `propagate()` method and
  `link_budget_summary()`

### `satellite_receiver.py`

- **`AGC`** – adaptive gain control towards unit RMS
- **`estimate_carrier_offset(signal, sample_rate)`** – 4th-power FFT estimator
- **`correct_carrier_offset(signal, offset_hz, sample_rate)`** – NCO correction
- **`qpsk_demodulate(symbols)`** – hard-decision nearest-neighbour demod
- **`compute_ber(tx_bits, rx_bits)`** – bit error rate
- **`compute_snr(signal, noise)`** – SNR in dB
- **`compute_evm(reference, received)`** – RMS EVM (%)
- **`SatelliteReceiver`** – convenience class wrapping the full RX chain

### `signal_analysis.py`

- **`plot_constellation(symbols)`** – I/Q scatter plot
- **`plot_eye_diagram(signal, sps)`** – overlaid symbol traces
- **`plot_psd(freqs, psd)`** – power spectrum in dB
- **`plot_ber_curve(snr_db, ber_sim)`** – simulated vs theoretical BER
- **`plot_analysis_dashboard(...)`** – four-panel composite figure
- **`theoretical_qpsk_ber(ebn0_db)`** – `0.5 · erfc(√(Eb/N0))`

### `config.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SYMBOL_RATE_HZ` | 1 Msps | Symbol rate |
| `SAMPLES_PER_SYMBOL` | 8 | Oversampling factor |
| `RC_ROLLOFF` | 0.35 | RRC excess bandwidth |
| `RC_FILTER_SPAN` | 10 | Filter span (symbol periods) |
| `CARRIER_FREQ_HZ` | 1 MHz | Simulation carrier frequency |

Orbit presets: **GEO** (35 786 km), **MEO** (10 000 km), **LEO** (550 km)

Frequency bands: **L**, **S**, **C**, **Ku**, **Ka**

---

## System Parameters (CLI)

| Argument | Default | Description |
|----------|---------|-------------|
| `--bits` | 10 000 | Number of information bits |
| `--snr-min` | 0 dB | Minimum Eb/N0 |
| `--snr-max` | 20 dB | Maximum Eb/N0 |
| `--orbit` | GEO | Satellite orbit |
| `--band` | Ku | RF frequency band |
| `--doppler` | off | Enable Doppler shift |
| `--seed` | 42 | Random seed |
| `--no-plot` | off | Suppress plots |

---

## Example Python API

```python
import numpy as np
from satellite_transmitter import SatelliteTransmitter, generate_bits
from propagation_channel import PropagationChannel
from satellite_receiver import SatelliteReceiver

rng = np.random.default_rng(42)

tx = SatelliteTransmitter()
bits = generate_bits(10_000, rng=rng)
tx_result = tx.transmit(bits)

channel = PropagationChannel(orbit="GEO", frequency_band="Ku")
ch_result = channel.propagate(
    tx_result["tx_signal"],
    snr_db=10.0,
    sample_rate=tx_result["sample_rate"],
    rng=rng,
)

rx = SatelliteReceiver()
rx_result = rx.receive(ch_result["received_signal"], tx_bits=bits)

print(f"BER  : {rx_result['ber']:.4e}")
print(f"EVM  : {rx_result['evm_pct']:.2f}%")
```

---

## Date Created
2026-04-30
