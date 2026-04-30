"""
Propagation Channel Model
=========================
Simulates the satellite-to-ground propagation effects:

  * Free-space path loss (FSPL)
  * Additive White Gaussian Noise (AWGN)
  * Doppler frequency shift

References
----------
MATLAB RF Satellite Link example:
  https://www.mathworks.com/help/comm/ug/rf-satellite-link.html
"""

import numpy as np

from config import (
    SPEED_OF_LIGHT,
    ORBIT_PRESETS,
    FREQUENCY_BANDS,
    CHANNEL_DEFAULTS,
)


# ---------------------------------------------------------------------------
# Helper: free-space path loss
# ---------------------------------------------------------------------------

def free_space_path_loss_db(
    distance_m: float,
    frequency_hz: float,
) -> float:
    """
    Friis free-space path loss in dB.

    FSPL = 20 log10(4π d f / c)

    Parameters
    ----------
    distance_m   : float   Link distance in metres.
    frequency_hz : float   Carrier frequency in Hz.

    Returns
    -------
    fspl_db : float  Path loss in dB (positive = loss).
    """
    if distance_m <= 0 or frequency_hz <= 0:
        raise ValueError("distance_m and frequency_hz must be positive.")
    fspl_linear = (4 * np.pi * distance_m * frequency_hz / SPEED_OF_LIGHT) ** 2
    return 10 * np.log10(fspl_linear)


def doppler_shift_hz(
    carrier_freq_hz: float,
    satellite_velocity_mps: float,
    elevation_angle_deg: float = 90.0,
) -> float:
    """
    Compute the Doppler frequency shift.

    Δf = f₀ · v · cos(θ) / c

    Parameters
    ----------
    carrier_freq_hz        : float  Carrier frequency in Hz.
    satellite_velocity_mps : float  Radial velocity (positive → approaching).
    elevation_angle_deg    : float  Elevation angle of the satellite (degrees).

    Returns
    -------
    delta_f : float  Doppler shift in Hz (positive → up-shift).
    """
    theta = np.deg2rad(elevation_angle_deg)
    return carrier_freq_hz * satellite_velocity_mps * np.cos(theta) / SPEED_OF_LIGHT


# ---------------------------------------------------------------------------
# Main channel class
# ---------------------------------------------------------------------------

class PropagationChannel:
    """
    Satellite propagation channel.

    Parameters
    ----------
    orbit           : str   Orbit preset key ('GEO', 'MEO', 'LEO').
    frequency_band  : str   Frequency band key ('Ku', 'Ka', 'S', …).
    noise_figure_db : float Receiver noise figure in dB.
    antenna_gain_db : float Receive antenna gain in dB.
    tx_power_dbw    : float Transmit power in dBW.
    tx_antenna_gain_db : float  Transmit antenna gain in dB.
    satellite_velocity_mps : float  Satellite velocity in m/s for Doppler.
    """

    def __init__(
        self,
        orbit: str = CHANNEL_DEFAULTS["orbit"],
        frequency_band: str = CHANNEL_DEFAULTS["frequency_band"],
        noise_figure_db: float = CHANNEL_DEFAULTS["noise_figure_db"],
        antenna_gain_db: float = CHANNEL_DEFAULTS["antenna_gain_db"],
        tx_power_dbw: float = CHANNEL_DEFAULTS["tx_power_dbw"],
        tx_antenna_gain_db: float = CHANNEL_DEFAULTS["tx_antenna_gain_db"],
        satellite_velocity_mps: float = CHANNEL_DEFAULTS["satellite_velocity_mps"],
    ):
        # Case-insensitive lookup for orbit
        orbit_key = next(
            (k for k in ORBIT_PRESETS if k.upper() == orbit.upper()), None
        )
        if orbit_key is None:
            raise ValueError(f"Unknown orbit preset '{orbit}'. "
                             f"Choose from {list(ORBIT_PRESETS)}.")
        orbit_info = ORBIT_PRESETS[orbit_key]

        # Case-insensitive lookup for frequency band
        band_key = next(
            (k for k in FREQUENCY_BANDS if k.upper() == frequency_band.upper()), None
        )
        if band_key is None:
            raise ValueError(f"Unknown frequency band '{frequency_band}'. "
                             f"Choose from {list(FREQUENCY_BANDS)}.")
        band_info = FREQUENCY_BANDS[band_key]

        self.orbit = orbit_key
        self.frequency_band = band_key
        self.altitude_m = orbit_info["altitude_m"]
        self.carrier_freq_hz = band_info["center_hz"]
        self.noise_figure_db = noise_figure_db
        self.antenna_gain_db = antenna_gain_db
        self.tx_power_dbw = tx_power_dbw
        self.tx_antenna_gain_db = tx_antenna_gain_db
        self.satellite_velocity_mps = satellite_velocity_mps

        # Derived
        self.fspl_db = free_space_path_loss_db(self.altitude_m, self.carrier_freq_hz)
        self.received_power_dbw = (
            tx_power_dbw
            + tx_antenna_gain_db
            - self.fspl_db
            + antenna_gain_db
        )

    # ------------------------------------------------------------------
    def add_awgn(
        self,
        signal: np.ndarray,
        snr_db: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Add AWGN to *signal* at the requested SNR (per sample).

        Parameters
        ----------
        signal : ndarray (real or complex)
        snr_db : float   Signal-to-noise ratio in dB (per sample).
        rng    : optional numpy Generator for reproducibility.

        Returns
        -------
        noisy_signal : ndarray, same shape as *signal*.
        """
        if rng is None:
            rng = np.random.default_rng()

        signal_power = np.mean(np.abs(signal) ** 2)
        if signal_power == 0:
            return signal.copy()

        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        sigma = np.sqrt(noise_power / 2)

        if np.iscomplexobj(signal):
            noise = sigma * (rng.standard_normal(signal.shape)
                             + 1j * rng.standard_normal(signal.shape))
        else:
            noise = sigma * rng.standard_normal(signal.shape)

        return signal + noise

    # ------------------------------------------------------------------
    def apply_doppler(
        self,
        signal: np.ndarray,
        sample_rate: float,
        elevation_angle_deg: float = 90.0,
    ) -> np.ndarray:
        """
        Apply a Doppler frequency shift to the signal.

        Parameters
        ----------
        signal              : ndarray (complex)
        sample_rate         : float  Sample rate in Hz.
        elevation_angle_deg : float  Elevation angle in degrees.

        Returns
        -------
        shifted_signal : ndarray, same shape as *signal*.
        """
        delta_f = doppler_shift_hz(
            self.carrier_freq_hz,
            self.satellite_velocity_mps,
            elevation_angle_deg,
        )
        t = np.arange(len(signal)) / sample_rate
        return signal * np.exp(1j * 2 * np.pi * delta_f * t)

    # ------------------------------------------------------------------
    def propagate(
        self,
        signal: np.ndarray,
        snr_db: float,
        sample_rate: float,
        apply_doppler: bool = False,
        elevation_angle_deg: float = 90.0,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """
        Apply the full channel model to a transmitted signal.

        Path loss is applied as a scalar amplitude scaling; AWGN is then
        added at the requested Eb/N0 (passed via *snr_db*).

        Parameters
        ----------
        signal              : ndarray  Transmitted waveform (complex baseband).
        snr_db              : float    Desired received SNR in dB.
        sample_rate         : float    Waveform sample rate in Hz.
        apply_doppler       : bool     Apply Doppler shift if True.
        elevation_angle_deg : float    Elevation angle for Doppler calc.
        rng                 : optional numpy Generator.

        Returns
        -------
        result : dict with keys
            'received_signal' – signal after channel effects
            'fspl_db'         – free-space path loss in dB
            'doppler_hz'      – Doppler shift applied (Hz)
            'snr_db'          – SNR at receiver (same as input)
        """
        # Path-loss amplitude scaling
        path_loss_linear = 10 ** (self.fspl_db / 20)
        received = signal / path_loss_linear

        # Doppler
        doppler_hz = 0.0
        if apply_doppler and self.satellite_velocity_mps != 0:
            received = self.apply_doppler(received, sample_rate, elevation_angle_deg)
            doppler_hz = doppler_shift_hz(
                self.carrier_freq_hz,
                self.satellite_velocity_mps,
                elevation_angle_deg,
            )

        # AWGN
        received = self.add_awgn(received, snr_db, rng=rng)

        return {
            "received_signal": received,
            "fspl_db": self.fspl_db,
            "doppler_hz": doppler_hz,
            "snr_db": snr_db,
        }

    # ------------------------------------------------------------------
    def link_budget_summary(self) -> dict:
        """Return a dictionary summarising the link budget."""
        return {
            "orbit":                  self.orbit,
            "frequency_band":         self.frequency_band,
            "altitude_m":             self.altitude_m,
            "carrier_freq_hz":        self.carrier_freq_hz,
            "tx_power_dbw":           self.tx_power_dbw,
            "tx_antenna_gain_db":     self.tx_antenna_gain_db,
            "fspl_db":                self.fspl_db,
            "rx_antenna_gain_db":     self.antenna_gain_db,
            "rx_noise_figure_db":     self.noise_figure_db,
            "received_power_dbw":     self.received_power_dbw,
        }
