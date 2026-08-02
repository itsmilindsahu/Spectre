"""
peaks.py

Feature extraction: detect peaks in a preprocessed spectrum and characterize
each one (position, height, width) so the rule engine / ML models have
something concrete to reason over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.signal import find_peaks, peak_widths

from spectre.ingestion.parsers import Spectrum


@dataclass
class Peak:
    wavenumber: float   # cm^-1, position of the peak
    intensity: float    # peak height (post-baseline-correction, normalized)
    width: float         # full width at half maximum, in cm^-1

    @property
    def shape(self) -> str:
        """Rough categorical shape label used by the rule engine."""
        if self.width >= 150:
            return "broad"
        if self.width >= 50:
            return "medium"
        return "sharp"


def detect_peaks(spectrum: Spectrum, height: float = 0.05,
                  prominence: float = 0.03, distance: int = 5) -> List[Peak]:
    """
    Detect peaks in a spectrum. Assumes `spectrum.intensities` represents
    absorbance-like data where peaks point UP (invert transmittance data
    before calling this if needed).

    Args:
        spectrum: a preprocessed (baseline-corrected, ideally normalized) Spectrum
        height: minimum peak height (post-normalization, so 0-1 scale) to count
        prominence: minimum prominence, filters out shoulders/noise
        distance: minimum distance between peaks, in samples

    Returns:
        List of Peak objects, sorted by wavenumber descending (matches
        spectrum convention: high energy / left side first).
    """
    y = spectrum.intensities
    x = spectrum.wavenumbers

    # find_peaks needs ascending-index data; our arrays are already indexed
    # consistently (x descending), so peak *indices* are valid regardless of
    # x direction -- we just need width results converted to cm^-1 using the
    # actual (possibly non-uniform) x spacing.
    idx, _ = find_peaks(y, height=height, prominence=prominence, distance=distance)

    if len(idx) == 0:
        return []

    widths_samples, _, _, _ = peak_widths(y, idx, rel_height=0.5)

    # Convert width in samples to width in cm^-1 using local spacing
    dx = np.abs(np.mean(np.diff(x))) if len(x) > 1 else 1.0

    peaks = [
        Peak(
            wavenumber=float(x[i]),
            intensity=float(y[i]),
            width=float(w * dx),
        )
        for i, w in zip(idx, widths_samples)
    ]

    peaks.sort(key=lambda p: p.wavenumber, reverse=True)
    return peaks
