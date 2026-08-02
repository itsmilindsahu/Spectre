"""
grid.py

Resample an arbitrary spectrum onto a fixed, common wavenumber grid so that
spectra from different instruments/sources become directly comparable, and
so downstream ML models get a fixed-length input vector.
"""

from __future__ import annotations

import numpy as np

from spectre.ingestion.parsers import Spectrum

DEFAULT_GRID = np.arange(4000, 399, -1.0)  # 4000 -> 400 cm^-1, 1 cm^-1 steps


def resample(spectrum: Spectrum, grid: np.ndarray = DEFAULT_GRID) -> Spectrum:
    """
    Linearly interpolate `spectrum` onto `grid`.

    np.interp requires ascending x, so we flip, interpolate, then flip back
    since Spectrum enforces descending order internally.
    """
    x = spectrum.wavenumbers[::-1]
    y = spectrum.intensities[::-1]
    grid_ascending = grid[::-1]

    y_interp = np.interp(grid_ascending, x, y, left=0.0, right=0.0)

    resampled = Spectrum(
        wavenumbers=grid_ascending[::-1].copy(),
        intensities=y_interp[::-1].copy(),
        metadata={**spectrum.metadata, "resampled": True},
    )
    return resampled


def normalize(spectrum: Spectrum) -> Spectrum:
    """Scale intensities to [0, 1] for comparability across spectra."""
    y = spectrum.intensities
    y_max = y.max() if y.max() > 0 else 1.0
    return Spectrum(
        wavenumbers=spectrum.wavenumbers.copy(),
        intensities=y / y_max,
        metadata={**spectrum.metadata, "normalized": True},
    )
