import numpy as np
import pytest

from spectre.ingestion.parsers import Spectrum
from spectre.features.peaks import detect_peaks


def _gaussian(x, center, width, height):
    return height * np.exp(-0.5 * ((x - center) / (width / 2.355)) ** 2)


def test_detect_single_sharp_peak():
    x = np.arange(4000, 399, -1.0)
    y = _gaussian(x, 1715, 12, 1.0)
    spectrum = Spectrum(wavenumbers=x, intensities=y)

    peaks = detect_peaks(spectrum, height=0.1, prominence=0.1)

    assert len(peaks) == 1
    assert abs(peaks[0].wavenumber - 1715) < 2
    assert peaks[0].shape == "sharp"


def test_detect_broad_peak_shape():
    x = np.arange(4000, 399, -1.0)
    y = _gaussian(x, 3400, 250, 0.8)
    spectrum = Spectrum(wavenumbers=x, intensities=y)

    peaks = detect_peaks(spectrum, height=0.1, prominence=0.1)

    assert len(peaks) == 1
    assert peaks[0].shape in ("broad", "medium")  # allow for FWHM measurement variance


def test_no_peaks_in_flat_spectrum():
    x = np.arange(4000, 399, -1.0)
    y = np.zeros_like(x)
    spectrum = Spectrum(wavenumbers=x, intensities=y)

    peaks = detect_peaks(spectrum, height=0.1, prominence=0.05)

    assert peaks == []


def test_spectrum_enforces_descending_order():
    x = np.arange(400, 4001, 1.0)  # ascending on purpose
    y = np.random.rand(len(x))
    spectrum = Spectrum(wavenumbers=x, intensities=y)

    assert spectrum.wavenumbers[0] > spectrum.wavenumbers[-1]
