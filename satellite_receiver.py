"""
Ground Station Downlink Receiver
=================================
Implements the signal-processing chain for a QPSK satellite receiver:

    received waveform
        → automatic gain control (AGC)
        → carrier-offset estimation & correction
        → matched filtering (RRC) + downsampling
        → QPSK demodulation (hard-decision)
        → BER, SNR, EVM metrics

References
----------
MATLAB RF Satellite Link example:
  https://www.mathworks.com/help/comm/ug/rf-satellite-link.html
"""

import numpy as np
from scipy.signal import upfirdn

from config import (
    BITS_PER_SYMBOL,
    SAMPLES_PER_SYMBOL,
    RC_ROLLOFF,
    RC_FILTER_SPAN,
    RECEIVER_DEFAULTS,
    SYMBOL_RATE_HZ,
)
from satellite_transmitter import _rrc_filter, _QPSK_SYMBOLS


# ---------------------------------------------------------------------------
# Automatic Gain Control
# ---------------------------------------------------------------------------

class AGC:
    """
    Simple adaptive automatic gain control.

    Adjusts the signal amplitude towards a target RMS of 1.0 using a
    first-order IIR feedback loop.

    Parameters
    ----------
    target_rms : float  Desired output RMS amplitude (default 1.0).
    rate       : float  Adaptation rate (0 < rate ≤ 1).  Smaller → slower.
    initial_gain_db : float  Starting gain in dB.
    """

    def __init__(
        self,
        target_rms: float = 1.0,
        rate: float = RECEIVER_DEFAULTS["agc_rate"],
        initial_gain_db: float = RECEIVER_DEFAULTS["agc_gain_db"],
    ):
        self.target_rms = target_rms
        self.rate = rate
        self.gain = 10 ** (initial_gain_db / 20)

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply AGC to *signal* and update the internal gain estimate.

        The gain is set so that the output RMS equals *target_rms*.  A
        first-order IIR smooths the gain across successive calls.

        Parameters
        ----------
        signal : ndarray (real or complex)

        Returns
        -------
        output : ndarray, same shape
        """
        # Measure current RMS
        current_rms = np.sqrt(np.mean(np.abs(signal) ** 2))
        if current_rms > 0:
            desired_gain = self.target_rms / current_rms
            # Smooth update (fast convergence when rate → 1)
            self.gain = (1 - self.rate) * self.gain + self.rate * desired_gain

        # Apply the updated gain directly so the output has target_rms
        current_rms2 = np.sqrt(np.mean(np.abs(signal) ** 2))
        if current_rms2 > 0:
            out_gain = self.target_rms / current_rms2
        else:
            out_gain = self.gain

        return signal * out_gain

    def reset(self, initial_gain_db: float = RECEIVER_DEFAULTS["agc_gain_db"]):
        """Reset gain to the specified value."""
        self.gain = 10 ** (initial_gain_db / 20)


# ---------------------------------------------------------------------------
# Carrier-offset estimation & correction
# ---------------------------------------------------------------------------

def estimate_carrier_offset(
    signal: np.ndarray,
    sample_rate: float,
    method: str = "fft",
) -> float:
    """
    Estimate carrier frequency offset using the squaring (M-th power) method.

    For QPSK, raising to the 4th power removes the modulation, leaving a
    tone at 4·Δf from which the offset is extracted.

    Parameters
    ----------
    signal      : complex ndarray  Received baseband signal.
    sample_rate : float            Sample rate in Hz.
    method      : str              Currently only 'fft' is supported.

    Returns
    -------
    offset_hz : float  Estimated frequency offset in Hz.
    """
    if method != "fft":
        raise NotImplementedError(f"Method '{method}' not implemented.")

    # Raise to 4th power to remove QPSK phase ambiguity
    s4 = signal ** 4
    spectrum = np.fft.fft(s4, n=len(s4))
    freqs = np.fft.fftfreq(len(s4), d=1.0 / sample_rate)
    peak_idx = np.argmax(np.abs(spectrum))
    return freqs[peak_idx] / 4.0


def correct_carrier_offset(
    signal: np.ndarray,
    offset_hz: float,
    sample_rate: float,
) -> np.ndarray:
    """
    Remove a carrier frequency offset from a complex baseband signal.

    Parameters
    ----------
    signal      : complex ndarray
    offset_hz   : float  Offset to correct (Hz).
    sample_rate : float  Sample rate in Hz.

    Returns
    -------
    corrected : complex ndarray, same shape as *signal*.
    """
    t = np.arange(len(signal)) / sample_rate
    return signal * np.exp(-1j * 2 * np.pi * offset_hz * t)


# ---------------------------------------------------------------------------
# QPSK demodulation
# ---------------------------------------------------------------------------

_QPSK_DEMOD_MAP = {0: np.array([0, 0]),
                   1: np.array([0, 1]),
                   2: np.array([1, 0]),
                   3: np.array([1, 1])}

# Direct reverse map: symbol index → bit pair
# Modulator: index = (b0 << 1) | b1, so b0 = (index >> 1) & 1, b1 = index & 1
_SYMBOL_TO_BITS = {
    idx: np.array([(idx >> 1) & 1, idx & 1], dtype=np.uint8)
    for idx in range(4)
}


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """
    Hard-decision QPSK demodulator.

    Finds the nearest constellation point (by Euclidean distance) and maps
    it back to the Gray-coded bit pair.

    Parameters
    ----------
    symbols : complex ndarray  Received (possibly noisy) baseband symbols.

    Returns
    -------
    bits : uint8 ndarray of shape (2 * len(symbols),).
    """
    # Compute distances to each constellation point
    dist = np.abs(symbols[:, None] - _QPSK_SYMBOLS[None, :])  # (N, 4)
    closest = np.argmin(dist, axis=1)                           # (N,)

    bits = np.empty(len(symbols) * 2, dtype=np.uint8)
    for i, idx in enumerate(closest):
        bits[2 * i:2 * i + 2] = _SYMBOL_TO_BITS[idx]
    return bits


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """
    Bit error rate.

    Parameters
    ----------
    tx_bits : ndarray of uint8
    rx_bits : ndarray of uint8  (length may differ; excess is ignored)

    Returns
    -------
    ber : float  Fraction of bits in error.
    """
    n = min(len(tx_bits), len(rx_bits))
    if n == 0:
        return float("nan")
    errors = np.sum(tx_bits[:n] != rx_bits[:n])
    return float(errors) / n


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Signal-to-noise ratio in dB given separate signal and noise arrays.

    Parameters
    ----------
    signal : ndarray
    noise  : ndarray

    Returns
    -------
    snr_db : float
    """
    s_power = np.mean(np.abs(signal) ** 2)
    n_power = np.mean(np.abs(noise) ** 2)
    if n_power == 0:
        return float("inf")
    return 10 * np.log10(s_power / n_power)


def compute_evm(
    reference_symbols: np.ndarray,
    received_symbols: np.ndarray,
) -> float:
    """
    RMS Error Vector Magnitude (EVM) as a percentage.

    EVM_rms = sqrt( mean(|e|²) / mean(|ref|²) ) × 100 %

    Parameters
    ----------
    reference_symbols : complex ndarray  Ideal (transmitted) symbols.
    received_symbols  : complex ndarray  Received (possibly noisy) symbols.

    Returns
    -------
    evm_pct : float
    """
    n = min(len(reference_symbols), len(received_symbols))
    error = received_symbols[:n] - reference_symbols[:n]
    ref_power = np.mean(np.abs(reference_symbols[:n]) ** 2)
    if ref_power == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.abs(error) ** 2) / ref_power) * 100)


# ---------------------------------------------------------------------------
# Main receiver class
# ---------------------------------------------------------------------------

class SatelliteReceiver:
    """
    Ground station QPSK receiver.

    Parameters
    ----------
    symbol_rate  : float  Symbol rate in symbols/second.
    sps          : int    Samples per symbol (oversampling factor).
    rolloff      : float  RRC roll-off factor (must match transmitter).
    filter_span  : int    RRC filter span in symbol periods.
    agc_rate     : float  AGC adaptation rate.
    """

    def __init__(
        self,
        symbol_rate: float = SYMBOL_RATE_HZ,
        sps: int = SAMPLES_PER_SYMBOL,
        rolloff: float = RC_ROLLOFF,
        filter_span: int = RC_FILTER_SPAN,
        agc_rate: float = RECEIVER_DEFAULTS["agc_rate"],
    ):
        self.symbol_rate = symbol_rate
        self.sps = sps
        self.rolloff = rolloff
        self.filter_span = filter_span
        self.sample_rate = symbol_rate * sps

        self.agc = AGC(rate=agc_rate)
        self.rrc_taps = _rrc_filter(rolloff, filter_span, sps)

    # ------------------------------------------------------------------
    def receive(
        self,
        rx_signal: np.ndarray,
        tx_bits: np.ndarray | None = None,
        carrier_offset_hz: float = 0.0,
        correct_offset: bool = True,
    ) -> dict:
        """
        Run the full receive chain.

        Parameters
        ----------
        rx_signal         : complex ndarray  Received baseband waveform.
        tx_bits           : optional ndarray  Original bits for BER/EVM.
        carrier_offset_hz : float             Known or estimated offset to correct.
        correct_offset    : bool              Apply carrier offset correction.

        Returns
        -------
        result : dict with keys
            'rx_bits'          – demodulated bits (uint8 ndarray)
            'rx_symbols'       – downsampled (matched-filtered) symbols
            'ber'              – BER (NaN if tx_bits not provided)
            'evm_pct'          – EVM % (NaN if tx_bits not provided)
            'estimated_offset' – carrier offset estimate (Hz)
            'agc_gain'         – final AGC gain (linear)
        """
        # 1. AGC
        signal = self.agc.apply(rx_signal)

        # 2. Carrier offset estimation
        estimated_offset = estimate_carrier_offset(signal, self.sample_rate)

        # 3. Carrier offset correction
        if correct_offset:
            # Use provided offset if nonzero, else use estimate
            offset_to_correct = carrier_offset_hz if carrier_offset_hz != 0 else estimated_offset
            signal = correct_carrier_offset(signal, offset_to_correct, self.sample_rate)

        # 4. Matched filtering (RRC) + downsampling
        # Apply RRC filter (same as transmitter — results in raised cosine overall)
        matched = upfirdn(self.rrc_taps, signal, down=self.sps)

        # Trim filter transient: filter delay = (span * sps) // 2 samples.
        # After downsampling the delay is filter_span // 2 symbols.
        delay_symbols = self.filter_span
        if delay_symbols < len(matched):
            rx_symbols = matched[delay_symbols:]
        else:
            rx_symbols = matched

        # 5. QPSK demodulation
        rx_bits = qpsk_demodulate(rx_symbols)

        # 6. Performance metrics
        ber = float("nan")
        evm_pct = float("nan")

        if tx_bits is not None:
            # Resolve QPSK phase ambiguity: try all 4 rotations, pick best BER
            best_ber = 1.0
            best_bits = rx_bits
            best_symbols = rx_symbols

            for k in range(4):
                rotated = rx_symbols * np.exp(1j * k * np.pi / 2)
                rot_bits = qpsk_demodulate(rotated)
                rot_ber = compute_ber(tx_bits, rot_bits)
                if rot_ber < best_ber:
                    best_ber = rot_ber
                    best_bits = rot_bits
                    best_symbols = rotated

            rx_bits = best_bits
            rx_symbols = best_symbols
            ber = best_ber

            # EVM against ideal QPSK constellation
            # Map tx_bits back to ideal symbols for comparison
            from satellite_transmitter import qpsk_modulate
            n_sym = min(len(rx_symbols), len(tx_bits) // 2)
            ideal_syms = qpsk_modulate(tx_bits[: n_sym * 2])
            evm_pct = compute_evm(ideal_syms, rx_symbols[:n_sym])

        return {
            "rx_bits":          rx_bits,
            "rx_symbols":       rx_symbols,
            "ber":              ber,
            "evm_pct":          evm_pct,
            "estimated_offset": estimated_offset,
            "agc_gain":         self.agc.gain,
        }
