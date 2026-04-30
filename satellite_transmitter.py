"""
Satellite Downlink Transmitter
==============================
Implements the signal-processing chain for a QPSK satellite transmitter:

    random bits
        → Gray-coded QPSK symbol mapping
        → raised-cosine pulse shaping (upsampling + RRC filter)
        → (optional) RF carrier modulation
        → output waveform

References
----------
MATLAB RF Satellite Link example:
  https://www.mathworks.com/help/comm/ug/rf-satellite-link.html
"""

import numpy as np
from scipy.signal import upfirdn, firwin, lfilter

from config import (
    BITS_PER_SYMBOL,
    SAMPLES_PER_SYMBOL,
    RC_ROLLOFF,
    RC_FILTER_SPAN,
    CARRIER_FREQ_HZ,
    SYMBOL_RATE_HZ,
)


# ---------------------------------------------------------------------------
# QPSK constellation  (Gray-coded, unit-energy)
# ---------------------------------------------------------------------------
_QPSK_MAP = {
    0b00: (1 + 1j) / np.sqrt(2),
    0b01: (-1 + 1j) / np.sqrt(2),
    0b11: (-1 - 1j) / np.sqrt(2),
    0b10: (1 - 1j) / np.sqrt(2),
}
_QPSK_SYMBOLS = np.array([_QPSK_MAP[k] for k in range(4)])


def generate_bits(num_bits: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a random binary array of length *num_bits*."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, 2, size=num_bits, dtype=np.uint8)


def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    Map pairs of bits to Gray-coded QPSK symbols.

    Parameters
    ----------
    bits : array_like of uint8
        Binary input.  Length must be even.

    Returns
    -------
    symbols : ndarray of complex128
        QPSK symbols, shape (len(bits) // 2,).
    """
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size % 2:
        raise ValueError("Number of bits must be even for QPSK.")
    pairs = bits.reshape(-1, 2)
    indices = (pairs[:, 0] << 1) | pairs[:, 1]
    return _QPSK_SYMBOLS[indices]


def _rrc_filter(rolloff: float, span: int, sps: int) -> np.ndarray:
    """
    Design a root-raised-cosine FIR filter.

    Parameters
    ----------
    rolloff : float   Excess-bandwidth factor α (0 < α ≤ 1).
    span    : int     Filter span in symbol periods.
    sps     : int     Samples per symbol.

    Returns
    -------
    h : ndarray of float  (length = span*sps + 1)
    """
    n_taps = span * sps + 1
    t = np.arange(n_taps) - (n_taps - 1) / 2  # centred time axis
    t = t / sps                                  # normalise to symbol periods

    h = np.zeros(n_taps)
    alpha = rolloff

    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = (1 + alpha * (4 / np.pi - 1))
        elif abs(ti) == 1 / (4 * alpha) and alpha != 0:
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            num = (
                np.sin(np.pi * ti * (1 - alpha))
                + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            )
            den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i] = num / den

    # Normalise to unit energy
    h /= np.sqrt(np.sum(h ** 2))
    return h


def pulse_shape(
    symbols: np.ndarray,
    sps: int = SAMPLES_PER_SYMBOL,
    rolloff: float = RC_ROLLOFF,
    span: int = RC_FILTER_SPAN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Upsample and apply root-raised-cosine pulse shaping.

    Parameters
    ----------
    symbols : ndarray of complex
    sps     : samples per symbol
    rolloff : RRC roll-off factor
    span    : filter span in symbol periods

    Returns
    -------
    tx_signal : ndarray of complex  Pulse-shaped waveform.
    rrc_taps  : ndarray of float    RRC filter coefficients.
    """
    h = _rrc_filter(rolloff, span, sps)
    # upfirdn: upsample by sps, convolve with h
    tx_signal = upfirdn(h, symbols, up=sps)
    return tx_signal, h


def modulate_rf(
    baseband: np.ndarray,
    carrier_freq: float = CARRIER_FREQ_HZ,
    sample_rate: float = SYMBOL_RATE_HZ * SAMPLES_PER_SYMBOL,
) -> np.ndarray:
    """
    Frequency-translate a complex baseband signal to a real passband signal.

    Parameters
    ----------
    baseband    : complex ndarray   Baseband waveform.
    carrier_freq: float             Carrier frequency in Hz.
    sample_rate : float             Sample rate in Hz.

    Returns
    -------
    rf_signal : real ndarray
    """
    t = np.arange(len(baseband)) / sample_rate
    carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
    return np.real(baseband * carrier)


def compute_psd(
    signal: np.ndarray,
    sample_rate: float,
    nfft: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the one-sided power spectral density via Welch's method.

    Parameters
    ----------
    signal      : ndarray   Input signal (real or complex).
    sample_rate : float     Sample rate in Hz.
    nfft        : int       FFT length.

    Returns
    -------
    freqs : ndarray   Frequency axis (Hz).
    psd   : ndarray   Power spectral density (linear).
    """
    from scipy.signal import welch

    nperseg = min(nfft, len(signal))
    freqs, psd = welch(signal, fs=sample_rate, nperseg=nperseg, nfft=nfft)
    return freqs, psd


class SatelliteTransmitter:
    """
    End-to-end satellite downlink transmitter.

    Parameters
    ----------
    symbol_rate   : float   Symbol rate in symbols/second.
    sps           : int     Samples per symbol (oversampling factor).
    rolloff       : float   RRC roll-off factor.
    filter_span   : int     RRC filter span in symbol periods.
    carrier_freq  : float   IF/RF carrier frequency in Hz.
    """

    def __init__(
        self,
        symbol_rate: float = SYMBOL_RATE_HZ,
        sps: int = SAMPLES_PER_SYMBOL,
        rolloff: float = RC_ROLLOFF,
        filter_span: int = RC_FILTER_SPAN,
        carrier_freq: float = CARRIER_FREQ_HZ,
    ):
        self.symbol_rate = symbol_rate
        self.sps = sps
        self.rolloff = rolloff
        self.filter_span = filter_span
        self.carrier_freq = carrier_freq
        self.sample_rate = symbol_rate * sps

        # Build matched-filter taps once
        self.rrc_taps = _rrc_filter(rolloff, filter_span, sps)

    # ------------------------------------------------------------------
    def transmit(
        self,
        bits: np.ndarray,
        apply_rf: bool = False,
    ) -> dict:
        """
        Run the full transmit chain.

        Parameters
        ----------
        bits     : ndarray of uint8   Information bits (length must be even).
        apply_rf : bool               If True, frequency-translate to RF.

        Returns
        -------
        result : dict with keys
            'bits'            – original bits
            'symbols'         – QPSK symbols (complex)
            'baseband_signal' – pulse-shaped baseband waveform (complex)
            'tx_signal'       – final transmitted signal
                                (real passband if apply_rf else complex baseband)
            'rrc_taps'        – RRC filter coefficients
            'sample_rate'     – waveform sample rate in Hz
        """
        symbols = qpsk_modulate(bits)
        baseband, _ = pulse_shape(symbols, self.sps, self.rolloff, self.filter_span)

        if apply_rf:
            tx_signal = modulate_rf(baseband, self.carrier_freq, self.sample_rate)
        else:
            tx_signal = baseband

        return {
            "bits": bits,
            "symbols": symbols,
            "baseband_signal": baseband,
            "tx_signal": tx_signal,
            "rrc_taps": self.rrc_taps,
            "sample_rate": self.sample_rate,
        }

    # ------------------------------------------------------------------
    def get_psd(self, signal: np.ndarray, nfft: int = 1024):
        """Return (freqs, psd) for *signal* at the transmitter sample rate."""
        return compute_psd(signal, self.sample_rate, nfft)
