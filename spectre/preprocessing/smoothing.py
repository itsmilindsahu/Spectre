"""
smoothing.py

Light smoothing applied after baseline correction and before peak detection.
Baseline correction (ALS) can leave small residual ripples, and raw
instrument data always carries some noise -- a short Savitzky-Golay filter
removes high-frequency noise while preserving peak shape/position, which is
important since peak width feeds directly into the rule engine's shape
classification (sharp/medium/broad).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def smooth(y: np.ndarray, window_length: int = 9, polyorder: int = 3) -> np.ndarray:
    """Apply a Savitzky-Golay filter. Falls back gracefully on short arrays."""
    wl = min(window_length, len(y) - (1 - len(y) % 2))
    if wl < polyorder + 2:
        return y  # too short to smooth meaningfully
    if wl % 2 == 0:
        wl -= 1
    return savgol_filter(y, window_length=wl, polyorder=polyorder)
