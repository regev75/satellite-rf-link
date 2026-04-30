"""
Signal Analysis and Visualization
==================================
Provides tools for plotting and analysing satellite link signals:

  * Constellation diagram
  * Eye diagram
  * Power spectral density (PSD) plot
  * BER vs Eb/N0 curve

References
----------
MATLAB RF Satellite Link example:
  https://www.mathworks.com/help/comm/ug/rf-satellite-link.html
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Constellation diagram
# ---------------------------------------------------------------------------

def plot_constellation(
    symbols: np.ndarray,
    title: str = "QPSK Constellation",
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a complex symbol constellation.

    Parameters
    ----------
    symbols : complex ndarray   Received/transmitted symbols.
    title   : str               Plot title.
    ax      : optional Axes     If provided, draw on this Axes.
    show    : bool              Call plt.show() if True.

    Returns
    -------
    fig : matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    ax.scatter(symbols.real, symbols.imag, s=3, alpha=0.4, color="steelblue")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Eye diagram
# ---------------------------------------------------------------------------

def plot_eye_diagram(
    signal: np.ndarray,
    sps: int,
    num_traces: int = 200,
    component: str = "real",
    title: str = "Eye Diagram",
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Overlay-plot an eye diagram from a pulse-shaped waveform.

    Parameters
    ----------
    signal     : ndarray (real or complex)
    sps        : int   Samples per symbol.
    num_traces : int   Number of symbol intervals to overlay.
    component  : str   'real' or 'imag'.
    title      : str   Plot title.
    ax         : optional Axes
    show       : bool

    Returns
    -------
    fig : matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    data = signal.real if component == "real" else signal.imag
    window = 2 * sps  # show 2 symbol periods
    n_traces = min(num_traces, len(data) // window)

    for i in range(n_traces):
        segment = data[i * sps: i * sps + window]
        if len(segment) == window:
            ax.plot(segment, color="steelblue", alpha=0.1, linewidth=0.8)

    ax.axvline(sps, color="red", linestyle="--", linewidth=0.8, label="Symbol boundary")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# PSD plot
# ---------------------------------------------------------------------------

def plot_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    title: str = "Power Spectral Density",
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a power spectral density estimate.

    Parameters
    ----------
    freqs : ndarray   Frequency axis (Hz).
    psd   : ndarray   PSD values (linear).
    title : str       Plot title.
    ax    : optional Axes
    show  : bool

    Returns
    -------
    fig : matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    psd_db = 10 * np.log10(np.maximum(psd, 1e-20))
    ax.plot(freqs / 1e3, psd_db, color="steelblue", linewidth=0.9)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# BER vs Eb/N0 curve
# ---------------------------------------------------------------------------

def theoretical_qpsk_ber(ebn0_db_range: np.ndarray) -> np.ndarray:
    """
    Theoretical QPSK BER = 0.5 * erfc(sqrt(Eb/N0)).

    Parameters
    ----------
    ebn0_db_range : ndarray  Eb/N0 values in dB.

    Returns
    -------
    ber_theory : ndarray  Theoretical BER values.
    """
    from scipy.special import erfc

    ebn0_linear = 10 ** (np.asarray(ebn0_db_range) / 10)
    return 0.5 * erfc(np.sqrt(ebn0_linear))


def plot_ber_curve(
    snr_db_values: list | np.ndarray,
    ber_simulated: list | np.ndarray,
    title: str = "BER vs Eb/N0",
    show_theory: bool = True,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot simulated (and optionally theoretical) BER vs Eb/N0.

    Parameters
    ----------
    snr_db_values  : array-like   Eb/N0 values in dB.
    ber_simulated  : array-like   Corresponding simulated BER values.
    title          : str
    show_theory    : bool         Overlay theoretical QPSK BER.
    ax             : optional Axes
    show           : bool

    Returns
    -------
    fig : matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    snr_arr = np.asarray(snr_db_values)
    ber_arr = np.asarray(ber_simulated, dtype=float)

    # Replace zero BER (can't plot on log scale) with NaN
    ber_arr[ber_arr == 0] = np.nan

    ax.semilogy(snr_arr, ber_arr, "o-", color="steelblue", label="Simulated")

    if show_theory:
        ber_theory = theoretical_qpsk_ber(snr_arr)
        ax.semilogy(snr_arr, ber_theory, "--", color="tomato", label="Theory (QPSK)")

    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Composite analysis dashboard
# ---------------------------------------------------------------------------

def plot_analysis_dashboard(
    tx_signal: np.ndarray,
    rx_symbols: np.ndarray,
    tx_freqs: np.ndarray,
    tx_psd: np.ndarray,
    snr_db_values: list | np.ndarray,
    ber_values: list | np.ndarray,
    sps: int,
    title: str = "RF Satellite Link Analysis",
    show: bool = True,
) -> plt.Figure:
    """
    Four-panel analysis dashboard:
      1. Transmitted PSD
      2. Transmitted eye diagram
      3. Received constellation
      4. BER vs Eb/N0

    Parameters
    ----------
    tx_signal      : ndarray   Transmitted waveform.
    rx_symbols     : ndarray   Received complex symbols.
    tx_freqs       : ndarray   PSD frequency axis (Hz).
    tx_psd         : ndarray   PSD values (linear).
    snr_db_values  : array-like
    ber_values     : array-like
    sps            : int       Samples per symbol.
    title          : str       Overall figure title.
    show           : bool

    Returns
    -------
    fig : matplotlib Figure
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax_psd   = fig.add_subplot(gs[0, 0])
    ax_eye   = fig.add_subplot(gs[0, 1])
    ax_const = fig.add_subplot(gs[1, 0])
    ax_ber   = fig.add_subplot(gs[1, 1])

    plot_psd(tx_freqs, tx_psd, title="Transmitted PSD", ax=ax_psd, show=False)
    plot_eye_diagram(tx_signal, sps, title="Eye Diagram (TX)", ax=ax_eye, show=False)
    plot_constellation(rx_symbols, title="Received Constellation", ax=ax_const, show=False)
    plot_ber_curve(snr_db_values, ber_values, title="BER vs Eb/N0", ax=ax_ber, show=False)

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Metrics summary helper
# ---------------------------------------------------------------------------

def compute_metrics_summary(
    tx_bits: np.ndarray,
    rx_bits: np.ndarray,
    rx_symbols: np.ndarray,
    tx_symbols: np.ndarray | None = None,
) -> dict:
    """
    Compute a dictionary of link performance metrics.

    Parameters
    ----------
    tx_bits   : ndarray of uint8
    rx_bits   : ndarray of uint8
    rx_symbols : complex ndarray
    tx_symbols : optional complex ndarray  (for EVM)

    Returns
    -------
    metrics : dict
        'ber'      – bit error rate
        'evm_pct'  – RMS EVM in % (NaN if tx_symbols not given)
        'n_bits'   – number of bits compared
        'n_errors' – number of bit errors
    """
    from satellite_receiver import compute_ber, compute_evm

    n = min(len(tx_bits), len(rx_bits))
    ber = compute_ber(tx_bits, rx_bits)
    n_errors = int(np.sum(tx_bits[:n] != rx_bits[:n]))

    evm_pct = float("nan")
    if tx_symbols is not None:
        evm_pct = compute_evm(tx_symbols, rx_symbols)

    return {
        "ber":     ber,
        "evm_pct": evm_pct,
        "n_bits":  n,
        "n_errors": n_errors,
    }
