"""
End-to-End RF Satellite Link Simulation
=========================================
Runs the complete transmitter → channel → receiver chain and produces
BER vs Eb/N0 performance curves together with signal analysis plots.

Usage
-----
    python simulate_link.py

Optional command-line arguments
--------------------------------
    --bits        Number of information bits  (default: 10000)
    --snr-min     Minimum Eb/N0 in dB        (default: 0)
    --snr-max     Maximum Eb/N0 in dB        (default: 20)
    --orbit       Satellite orbit preset     (default: GEO)
    --band        RF frequency band          (default: Ku)
    --doppler     Enable Doppler simulation  (flag)
    --seed        Random seed                (default: 42)
    --no-plot     Suppress all plots         (flag)
"""

import argparse
import numpy as np

from config import SIMULATION_DEFAULTS, ORBIT_PRESETS, FREQUENCY_BANDS
from satellite_transmitter import SatelliteTransmitter, generate_bits
from propagation_channel import PropagationChannel
from satellite_receiver import SatelliteReceiver
from signal_analysis import (
    plot_analysis_dashboard,
    plot_ber_curve,
    theoretical_qpsk_ber,
)


# ---------------------------------------------------------------------------
# Single-point simulation
# ---------------------------------------------------------------------------

def run_link(
    num_bits: int,
    snr_db: float,
    orbit: str = "GEO",
    frequency_band: str = "Ku",
    apply_doppler: bool = False,
    seed: int = 42,
) -> dict:
    """
    Simulate one operating point (single SNR value) of the satellite link.

    Parameters
    ----------
    num_bits        : int   Number of information bits to transmit.
    snr_db          : float Eb/N0 at the receiver input in dB.
    orbit           : str   Satellite orbit ('GEO', 'MEO', 'LEO').
    frequency_band  : str   RF band ('Ku', 'Ka', 'S', …).
    apply_doppler   : bool  Apply Doppler shift in the channel.
    seed            : int   Random seed.

    Returns
    -------
    result : dict
        'ber'          – simulated bit error rate
        'evm_pct'      – Error Vector Magnitude in %
        'tx_result'    – transmitter output dict
        'rx_result'    – receiver output dict
        'channel_info' – channel dict
    """
    rng = np.random.default_rng(seed)

    # Transmitter
    tx = SatelliteTransmitter()
    bits = generate_bits(num_bits, rng=rng)
    tx_result = tx.transmit(bits)

    # Channel
    channel = PropagationChannel(orbit=orbit, frequency_band=frequency_band)
    ch_result = channel.propagate(
        tx_result["tx_signal"],
        snr_db=snr_db,
        sample_rate=tx_result["sample_rate"],
        apply_doppler=apply_doppler,
        rng=rng,
    )

    # Receiver
    rx = SatelliteReceiver()
    rx_result = rx.receive(
        ch_result["received_signal"],
        tx_bits=bits,
    )

    return {
        "ber":          rx_result["ber"],
        "evm_pct":      rx_result["evm_pct"],
        "tx_result":    tx_result,
        "rx_result":    rx_result,
        "channel_info": ch_result,
    }


# ---------------------------------------------------------------------------
# BER sweep
# ---------------------------------------------------------------------------

def run_ber_sweep(
    snr_range_db: list,
    num_bits: int = SIMULATION_DEFAULTS["num_bits"],
    orbit: str = "GEO",
    frequency_band: str = "Ku",
    apply_doppler: bool = False,
    seed: int = SIMULATION_DEFAULTS["random_seed"],
    verbose: bool = True,
) -> tuple[list, list]:
    """
    Sweep Eb/N0 values and return simulated BER at each point.

    Parameters
    ----------
    snr_range_db   : list of float  Eb/N0 values to simulate.
    num_bits       : int
    orbit          : str
    frequency_band : str
    apply_doppler  : bool
    seed           : int
    verbose        : bool           Print progress.

    Returns
    -------
    snr_values : list of float
    ber_values : list of float
    """
    ber_values = []

    for i, snr in enumerate(snr_range_db):
        result = run_link(
            num_bits=num_bits,
            snr_db=snr,
            orbit=orbit,
            frequency_band=frequency_band,
            apply_doppler=apply_doppler,
            seed=seed + i,
        )
        ber = result["ber"]
        ber_values.append(ber)

        if verbose:
            print(f"  Eb/N0 = {snr:5.1f} dB | BER = {ber:.4e} | EVM = {result['evm_pct']:.2f}%")

    return list(snr_range_db), ber_values


# ---------------------------------------------------------------------------
# Link budget report
# ---------------------------------------------------------------------------

def print_link_budget(orbit: str = "GEO", frequency_band: str = "Ku") -> None:
    """Print a formatted link budget table."""
    channel = PropagationChannel(orbit=orbit, frequency_band=frequency_band)
    budget = channel.link_budget_summary()

    print("\n" + "=" * 55)
    print("  RF Satellite Link Budget Summary")
    print("=" * 55)
    print(f"  Orbit            : {budget['orbit']}")
    print(f"  Frequency band   : {budget['frequency_band']}")
    print(f"  Altitude         : {budget['altitude_m'] / 1e3:.0f} km")
    print(f"  Carrier freq     : {budget['carrier_freq_hz'] / 1e9:.1f} GHz")
    print(f"  TX power         : {budget['tx_power_dbw']:.1f} dBW")
    print(f"  TX antenna gain  : {budget['tx_antenna_gain_db']:.1f} dB")
    print(f"  Free-space loss  : {budget['fspl_db']:.1f} dB")
    print(f"  RX antenna gain  : {budget['rx_antenna_gain_db']:.1f} dB")
    print(f"  RX noise figure  : {budget['rx_noise_figure_db']:.1f} dB")
    print(f"  Received power   : {budget['received_power_dbw']:.1f} dBW")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RF Satellite Link Simulation")
    parser.add_argument("--bits",     type=int,   default=SIMULATION_DEFAULTS["num_bits"])
    parser.add_argument("--snr-min",  type=float, default=0)
    parser.add_argument("--snr-max",  type=float, default=20)
    parser.add_argument("--orbit",    type=str,   default="GEO",
                        choices=list(ORBIT_PRESETS))
    parser.add_argument("--band",     type=str,   default="Ku",
                        choices=list(FREQUENCY_BANDS))
    parser.add_argument("--doppler",  action="store_true")
    parser.add_argument("--seed",     type=int,   default=SIMULATION_DEFAULTS["random_seed"])
    parser.add_argument("--no-plot",  action="store_true")
    args = parser.parse_args()

    snr_range = list(range(int(args.snr_min), int(args.snr_max) + 1, 2))

    print(f"\nRF Satellite Link Simulation")
    print(f"  Orbit: {args.orbit}  |  Band: {args.band}  |  Bits: {args.bits}")
    print(f"  Eb/N0 range: {args.snr_min}–{args.snr_max} dB  |  Doppler: {args.doppler}\n")

    print_link_budget(args.orbit, args.band)

    print("Running BER sweep …")
    snr_vals, ber_vals = run_ber_sweep(
        snr_range_db=snr_range,
        num_bits=args.bits,
        orbit=args.orbit,
        frequency_band=args.band,
        apply_doppler=args.doppler,
        seed=args.seed,
    )

    # --- Summary ---
    print("\n" + "-" * 45)
    print(f"{'Eb/N0 (dB)':>12}  {'BER':>12}  {'Theory':>12}")
    print("-" * 45)
    theory = theoretical_qpsk_ber(np.array(snr_vals))
    for s, b, t in zip(snr_vals, ber_vals, theory):
        print(f"{s:12.1f}  {b:12.4e}  {t:12.4e}")
    print("-" * 45)

    if not args.no_plot:
        # Run one operating point for detailed plots
        mid_snr = snr_range[len(snr_range) // 2]
        detail = run_link(
            num_bits=args.bits,
            snr_db=mid_snr,
            orbit=args.orbit,
            frequency_band=args.band,
            apply_doppler=args.doppler,
            seed=args.seed,
        )
        tx = SatelliteTransmitter()
        freqs, psd = tx.get_psd(detail["tx_result"]["tx_signal"])

        plot_analysis_dashboard(
            tx_signal=detail["tx_result"]["baseband_signal"],
            rx_symbols=detail["rx_result"]["rx_symbols"],
            tx_freqs=freqs,
            tx_psd=psd,
            snr_db_values=snr_vals,
            ber_values=ber_vals,
            sps=tx.sps,
            title=f"RF Satellite Link Analysis — {args.orbit} / {args.band}-band  "
                  f"(Eb/N0={mid_snr} dB)",
        )


if __name__ == "__main__":
    main()
