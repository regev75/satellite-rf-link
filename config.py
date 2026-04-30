"""
Configuration parameters for the RF Satellite Link simulation.

Defines system-level, satellite, channel, transmitter, receiver, and
simulation parameters. All physical constants and default values are
collected here so they can be tuned from a single location.
"""

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
SPEED_OF_LIGHT = 3.0e8  # m/s

# ---------------------------------------------------------------------------
# Satellite orbit presets  (altitude in metres)
# ---------------------------------------------------------------------------
ORBIT_PRESETS = {
    "GEO": {"altitude_m": 35_786_000, "description": "Geostationary orbit"},
    "MEO": {"altitude_m": 10_000_000, "description": "Medium Earth orbit"},
    "LEO": {"altitude_m": 550_000,    "description": "Low Earth orbit"},
}

# ---------------------------------------------------------------------------
# RF frequency band presets  (centre frequency in Hz)
# ---------------------------------------------------------------------------
FREQUENCY_BANDS = {
    "L":  {"center_hz": 1.5e9,  "description": "L-band  (1–2 GHz)"},
    "S":  {"center_hz": 2.5e9,  "description": "S-band  (2–4 GHz)"},
    "C":  {"center_hz": 6.0e9,  "description": "C-band  (4–8 GHz)"},
    "Ku": {"center_hz": 12.0e9, "description": "Ku-band (12–18 GHz)"},
    "Ka": {"center_hz": 20.0e9, "description": "Ka-band (26.5–40 GHz)"},
}

# ---------------------------------------------------------------------------
# Default system parameters
# ---------------------------------------------------------------------------

# Modulation
MODULATION = "QPSK"           # modulation scheme
BITS_PER_SYMBOL = 2           # QPSK → 2 bits/symbol

# Symbol rate and sampling
SYMBOL_RATE_HZ = 1e6          # 1 Msps
SAMPLES_PER_SYMBOL = 8        # oversampling factor (upsampled rate)

# Raised-cosine pulse shaping
RC_ROLLOFF = 0.35             # excess bandwidth factor (0 < α ≤ 1)
RC_FILTER_SPAN = 10           # filter span in symbol periods

# Carrier frequency (used when translating to RF in the transmitter)
CARRIER_FREQ_HZ = 1.0e6       # baseband-to-IF carrier (for simulation)

# ---------------------------------------------------------------------------
# Propagation channel defaults
# ---------------------------------------------------------------------------
CHANNEL_DEFAULTS = {
    "orbit":           "GEO",
    "frequency_band":  "Ku",
    "satellite_velocity_mps": 0.0,    # m/s  (non-zero → Doppler)
    "noise_figure_db": 3.0,           # receiver noise figure
    "antenna_gain_db": 40.0,          # receive antenna gain
    "tx_power_dbw":    10.0,          # transmit power in dBW
    "tx_antenna_gain_db": 37.0,       # transmit antenna gain
}

# ---------------------------------------------------------------------------
# Receiver defaults
# ---------------------------------------------------------------------------
RECEIVER_DEFAULTS = {
    "agc_gain_db":        20.0,   # initial AGC gain  (dB)
    "agc_rate":           0.01,   # AGC adaptation rate
    "carrier_offset_hz":  1000.0, # initial carrier freq offset for testing
    "pll_bandwidth_hz":   100.0,  # PLL loop bandwidth
    "timing_offset":      0.0,    # fractional timing offset (symbols)
}

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------
SIMULATION_DEFAULTS = {
    "num_bits":        10_000,             # number of information bits
    "snr_range_db":    list(range(0, 21)), # Eb/N0 values for BER sweep
    "random_seed":     42,
    "apply_doppler":   False,
    "apply_fading":    False,
}
