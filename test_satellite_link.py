"""
Unit tests for the RF Satellite Link system.

Run with:
    pytest test_satellite_link.py -v
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_orbit_presets_present(self):
        from config import ORBIT_PRESETS
        for key in ("GEO", "MEO", "LEO"):
            assert key in ORBIT_PRESETS
            assert "altitude_m" in ORBIT_PRESETS[key]

    def test_frequency_bands_present(self):
        from config import FREQUENCY_BANDS
        for key in ("L", "S", "C", "Ku", "Ka"):
            assert key in FREQUENCY_BANDS
            assert "center_hz" in FREQUENCY_BANDS[key]

    def test_simulation_defaults(self):
        from config import SIMULATION_DEFAULTS
        assert SIMULATION_DEFAULTS["num_bits"] > 0
        assert len(SIMULATION_DEFAULTS["snr_range_db"]) > 0


# ---------------------------------------------------------------------------
# Transmitter
# ---------------------------------------------------------------------------

class TestTransmitter:
    def test_generate_bits_length(self):
        from satellite_transmitter import generate_bits
        bits = generate_bits(100)
        assert len(bits) == 100

    def test_generate_bits_binary(self):
        from satellite_transmitter import generate_bits
        bits = generate_bits(1000, rng=np.random.default_rng(0))
        assert set(bits.tolist()).issubset({0, 1})

    def test_qpsk_modulate_output_shape(self):
        from satellite_transmitter import qpsk_modulate
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        syms = qpsk_modulate(bits)
        assert len(syms) == 4

    def test_qpsk_modulate_unit_magnitude(self):
        from satellite_transmitter import qpsk_modulate
        # 4 symbols, cycling through all QPSK points
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        syms = qpsk_modulate(bits)
        np.testing.assert_allclose(np.abs(syms), 1.0, atol=1e-10)

    def test_qpsk_modulate_odd_bits_raises(self):
        from satellite_transmitter import qpsk_modulate
        with pytest.raises(ValueError):
            qpsk_modulate(np.array([0, 1, 0], dtype=np.uint8))

    def test_rrc_filter_length(self):
        from satellite_transmitter import _rrc_filter
        sps, span = 8, 10
        h = _rrc_filter(0.35, span, sps)
        assert len(h) == span * sps + 1

    def test_rrc_filter_unit_energy(self):
        from satellite_transmitter import _rrc_filter
        h = _rrc_filter(0.35, 10, 8)
        np.testing.assert_allclose(np.sum(h ** 2), 1.0, atol=1e-6)

    def test_pulse_shape_output_length(self):
        from satellite_transmitter import pulse_shape, qpsk_modulate
        bits = np.zeros(100, dtype=np.uint8)
        syms = qpsk_modulate(bits)
        sps, span = 8, 10
        tx, h = pulse_shape(syms, sps=sps, span=span)
        # scipy upfirdn output length: (len(x) - 1) * up + len(h)
        expected = (len(syms) - 1) * sps + len(h)
        assert len(tx) == expected

    def test_transmitter_end_to_end(self):
        from satellite_transmitter import SatelliteTransmitter, generate_bits
        tx = SatelliteTransmitter()
        bits = generate_bits(200, rng=np.random.default_rng(0))
        result = tx.transmit(bits)
        assert "tx_signal" in result
        assert len(result["tx_signal"]) > 0
        assert np.iscomplexobj(result["tx_signal"])

    def test_transmitter_rf_output_real(self):
        from satellite_transmitter import SatelliteTransmitter, generate_bits
        tx = SatelliteTransmitter()
        bits = generate_bits(200, rng=np.random.default_rng(1))
        result = tx.transmit(bits, apply_rf=True)
        assert np.isrealobj(result["tx_signal"])

    def test_psd_output(self):
        from satellite_transmitter import SatelliteTransmitter, generate_bits
        tx = SatelliteTransmitter()
        bits = generate_bits(500, rng=np.random.default_rng(2))
        result = tx.transmit(bits)
        freqs, psd = tx.get_psd(result["tx_signal"])
        assert len(freqs) == len(psd)
        assert np.all(psd >= 0)


# ---------------------------------------------------------------------------
# Propagation channel
# ---------------------------------------------------------------------------

class TestPropagationChannel:
    def test_fspl_increases_with_distance(self):
        from propagation_channel import free_space_path_loss_db
        fspl1 = free_space_path_loss_db(1_000_000, 12e9)
        fspl2 = free_space_path_loss_db(2_000_000, 12e9)
        assert fspl2 > fspl1

    def test_fspl_increases_with_frequency(self):
        from propagation_channel import free_space_path_loss_db
        fspl1 = free_space_path_loss_db(36_786_000, 12e9)
        fspl2 = free_space_path_loss_db(36_786_000, 20e9)
        assert fspl2 > fspl1

    def test_fspl_invalid_inputs(self):
        from propagation_channel import free_space_path_loss_db
        with pytest.raises(ValueError):
            free_space_path_loss_db(0, 12e9)
        with pytest.raises(ValueError):
            free_space_path_loss_db(1e6, 0)

    def test_doppler_zero_velocity(self):
        from propagation_channel import doppler_shift_hz
        assert doppler_shift_hz(12e9, 0.0) == 0.0

    def test_doppler_positive_velocity(self):
        from propagation_channel import doppler_shift_hz
        df = doppler_shift_hz(12e9, 1000.0)
        assert df > 0

    def test_channel_awgn_power(self):
        from propagation_channel import PropagationChannel
        ch = PropagationChannel()
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(10_000) + 1j * rng.standard_normal(10_000)
        noisy = ch.add_awgn(signal, snr_db=20.0, rng=rng)
        assert noisy.shape == signal.shape

    def test_channel_awgn_snr(self):
        """Measured SNR should be within ±3 dB of requested."""
        from propagation_channel import PropagationChannel
        ch = PropagationChannel()
        rng = np.random.default_rng(0)
        n = 50_000
        signal = np.ones(n, dtype=complex)
        target_snr = 15.0
        noisy = ch.add_awgn(signal, snr_db=target_snr, rng=rng)
        noise = noisy - signal
        snr_meas = 10 * np.log10(
            np.mean(np.abs(signal) ** 2) / np.mean(np.abs(noise) ** 2)
        )
        assert abs(snr_meas - target_snr) < 3.0

    def test_channel_propagate_output_keys(self):
        from propagation_channel import PropagationChannel
        ch = PropagationChannel()
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(1000) + 1j * rng.standard_normal(1000)
        result = ch.propagate(sig, snr_db=10.0, sample_rate=8e6, rng=rng)
        for key in ("received_signal", "fspl_db", "doppler_hz", "snr_db"):
            assert key in result

    def test_channel_invalid_orbit(self):
        from propagation_channel import PropagationChannel
        with pytest.raises(ValueError):
            PropagationChannel(orbit="INVALID")

    def test_channel_invalid_band(self):
        from propagation_channel import PropagationChannel
        with pytest.raises(ValueError):
            PropagationChannel(frequency_band="INVALID")

    def test_link_budget_summary_keys(self):
        from propagation_channel import PropagationChannel
        budget = PropagationChannel().link_budget_summary()
        for key in ("orbit", "fspl_db", "received_power_dbw"):
            assert key in budget


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------

class TestReceiver:
    def _make_rx_input(self, n_bits=500, snr_db=20.0, seed=0):
        """Helper: transmit bits through channel and return (rx_signal, bits)."""
        from satellite_transmitter import SatelliteTransmitter, generate_bits
        from propagation_channel import PropagationChannel
        rng = np.random.default_rng(seed)
        tx = SatelliteTransmitter()
        bits = generate_bits(n_bits, rng=rng)
        tx_res = tx.transmit(bits)
        ch = PropagationChannel()
        ch_res = ch.propagate(tx_res["tx_signal"], snr_db=snr_db,
                               sample_rate=tx_res["sample_rate"], rng=rng)
        return ch_res["received_signal"], bits, tx_res["sample_rate"]

    def test_agc_normalizes_amplitude(self):
        from satellite_receiver import AGC
        agc = AGC()
        rng = np.random.default_rng(0)
        signal = 100 * (rng.standard_normal(1000) + 1j * rng.standard_normal(1000))
        out = agc.apply(signal)
        rms = np.sqrt(np.mean(np.abs(out) ** 2))
        assert rms < 10  # amplitude should be reduced significantly

    def test_carrier_offset_estimation(self):
        from satellite_receiver import estimate_carrier_offset, correct_carrier_offset
        from satellite_transmitter import SatelliteTransmitter, generate_bits
        tx = SatelliteTransmitter()
        bits = generate_bits(1000, rng=np.random.default_rng(0))
        tx_res = tx.transmit(bits)
        sig = tx_res["tx_signal"]
        sr = tx_res["sample_rate"]
        # Apply a known offset
        t = np.arange(len(sig)) / sr
        known_offset = 500.0
        sig_offset = sig * np.exp(1j * 2 * np.pi * known_offset * t)
        est = estimate_carrier_offset(sig_offset, sr)
        # Tolerance: resolution = sr / len(sig)
        tol = sr / len(sig) * 4
        assert abs(est - known_offset) < tol or abs(est + known_offset) < tol

    def test_receiver_output_keys(self):
        from satellite_receiver import SatelliteReceiver
        rx_sig, bits, _ = self._make_rx_input(snr_db=20.0)
        rx = SatelliteReceiver()
        result = rx.receive(rx_sig, tx_bits=bits)
        for key in ("rx_bits", "rx_symbols", "ber", "evm_pct",
                    "estimated_offset", "agc_gain"):
            assert key in result

    def test_ber_high_snr(self):
        """At 20 dB SNR, BER should be very low."""
        from satellite_receiver import SatelliteReceiver
        rx_sig, bits, _ = self._make_rx_input(n_bits=1000, snr_db=20.0, seed=7)
        rx = SatelliteReceiver()
        result = rx.receive(rx_sig, tx_bits=bits)
        assert result["ber"] < 0.05

    def test_ber_low_snr_higher_than_high_snr(self):
        """BER at 0 dB should be ≥ BER at 15 dB."""
        from satellite_receiver import SatelliteReceiver
        rx_sig_lo, bits_lo, _ = self._make_rx_input(n_bits=2000, snr_db=0.0, seed=3)
        rx_sig_hi, bits_hi, _ = self._make_rx_input(n_bits=2000, snr_db=15.0, seed=3)
        rx = SatelliteReceiver()
        ber_lo = rx.receive(rx_sig_lo, tx_bits=bits_lo)["ber"]
        ber_hi = rx.receive(rx_sig_hi, tx_bits=bits_hi)["ber"]
        assert ber_lo >= ber_hi

    def test_compute_ber(self):
        from satellite_receiver import compute_ber
        tx = np.array([0, 1, 0, 1], dtype=np.uint8)
        rx = np.array([0, 1, 1, 1], dtype=np.uint8)
        assert compute_ber(tx, rx) == pytest.approx(0.25)

    def test_compute_ber_perfect(self):
        from satellite_receiver import compute_ber
        bits = np.zeros(100, dtype=np.uint8)
        assert compute_ber(bits, bits) == 0.0

    def test_compute_evm(self):
        from satellite_receiver import compute_evm
        ref = np.array([1+1j, -1+1j]) / np.sqrt(2)
        # No error → EVM = 0
        np.testing.assert_allclose(compute_evm(ref, ref), 0.0, atol=1e-8)

    def test_compute_evm_nonzero(self):
        from satellite_receiver import compute_evm
        ref = np.array([1+1j, -1+1j]) / np.sqrt(2)
        noisy = ref + 0.1 * (1 + 1j) / np.sqrt(2)
        assert compute_evm(ref, noisy) > 0


# ---------------------------------------------------------------------------
# Signal analysis (no display)
# ---------------------------------------------------------------------------

class TestSignalAnalysis:
    def test_theoretical_ber_decreasing(self):
        from signal_analysis import theoretical_qpsk_ber
        snr = np.array([0, 5, 10, 15, 20])
        ber = theoretical_qpsk_ber(snr)
        assert np.all(np.diff(ber) < 0)

    def test_theoretical_ber_range(self):
        from signal_analysis import theoretical_qpsk_ber
        ber = theoretical_qpsk_ber(np.array([0.0]))
        assert 0 < ber[0] < 0.5

    def test_plot_constellation_no_display(self):
        """Ensure plotting functions run without errors (display suppressed)."""
        import matplotlib
        matplotlib.use("Agg")
        from signal_analysis import plot_constellation
        syms = (np.random.default_rng(0).standard_normal(100)
                + 1j * np.random.default_rng(1).standard_normal(100))
        fig = plot_constellation(syms, show=False)
        assert fig is not None

    def test_plot_psd_no_display(self):
        import matplotlib
        matplotlib.use("Agg")
        from signal_analysis import plot_psd
        freqs = np.linspace(0, 1e6, 512)
        psd = np.ones(512)
        fig = plot_psd(freqs, psd, show=False)
        assert fig is not None

    def test_plot_ber_no_display(self):
        import matplotlib
        matplotlib.use("Agg")
        from signal_analysis import plot_ber_curve
        snr = [0, 5, 10, 15]
        ber = [0.1, 0.01, 0.001, 0.0001]
        fig = plot_ber_curve(snr, ber, show=False)
        assert fig is not None

    def test_plot_eye_no_display(self):
        import matplotlib
        matplotlib.use("Agg")
        from signal_analysis import plot_eye_diagram
        signal = np.random.default_rng(0).standard_normal(1000)
        fig = plot_eye_diagram(signal, sps=8, show=False)
        assert fig is not None

    def test_compute_metrics_summary(self):
        from signal_analysis import compute_metrics_summary
        tx_bits = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
        rx_bits = tx_bits.copy()
        syms = np.ones(4, dtype=complex)
        m = compute_metrics_summary(tx_bits, rx_bits, syms)
        assert m["ber"] == 0.0
        assert m["n_errors"] == 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_link_low_ber_at_high_snr(self):
        """Full chain: BER at 15 dB Eb/N0 should be < 5%."""
        from simulate_link import run_link
        result = run_link(num_bits=2000, snr_db=15.0, seed=0)
        assert result["ber"] < 0.05

    def test_full_link_result_keys(self):
        from simulate_link import run_link
        result = run_link(num_bits=500, snr_db=10.0, seed=1)
        for key in ("ber", "evm_pct", "tx_result", "rx_result", "channel_info"):
            assert key in result

    def test_ber_sweep_length(self):
        from simulate_link import run_ber_sweep
        snr_range = [0, 5, 10]
        snr_vals, ber_vals = run_ber_sweep(
            snr_range_db=snr_range, num_bits=500, verbose=False
        )
        assert len(snr_vals) == 3
        assert len(ber_vals) == 3

    def test_ber_sweep_monotone(self):
        """BER should generally decrease as SNR increases."""
        from simulate_link import run_ber_sweep
        snr_range = [0, 5, 10, 15]
        _, ber_vals = run_ber_sweep(
            snr_range_db=snr_range, num_bits=1000, verbose=False, seed=99
        )
        # Allow at most one non-monotone step (statistical fluctuation)
        non_mono = sum(1 for a, b in zip(ber_vals, ber_vals[1:]) if b > a)
        assert non_mono <= 1
