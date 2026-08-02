"""Tests for spectre.ingestion.image_digitizer -- renders a known synthetic
spectrum to PNG, then checks the digitizer recovers it and that the
recovered spectrum still classifies correctly through the rule engine."""

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectre.ingestion.image_digitizer import (
    digitize_plot, calibrate_and_save, detect_plot_bbox, detect_curve_color, _load_rgb,
)
from spectre.preprocessing.baseline import correct_baseline
from spectre.preprocessing.grid import normalize
from spectre.preprocessing.smoothing import smooth
from spectre.features.peaks import detect_peaks
from spectre.models.rule_engine import classify
from spectre.synthetic.generator import generate_spectrum, GRID


def _render_test_plot(path, active_groups, seed=1):
    rng = np.random.default_rng(seed)
    y = generate_spectrum(rng, active_groups=active_groups, dropout_prob=0.0,
                           decoy_prob=0.0, noise_level=0.005)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(GRID, y, color="tab:blue", linewidth=1.5)
    ax.set_xlim(4000, 400)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return y


def test_detect_bbox_and_curve_color(tmp_path):
    img_path = tmp_path / "plot.png"
    _render_test_plot(img_path, {"Carbonyl (Ketone / Aldehyde / Ester / Carboxylic Acid)"})

    rgb = _load_rgb(str(img_path))
    bbox = detect_plot_bbox(rgb)
    left, top, right, bottom = bbox
    assert right > left and bottom > top
    # bbox should occupy a substantial chunk of the image, not a stray line
    assert (right - left) > rgb.shape[1] * 0.5
    assert (bottom - top) > rgb.shape[0] * 0.5

    color = detect_curve_color(rgb, bbox)
    assert color != (255, 255, 255)  # not background white


def test_digitize_plot_recovers_known_spectrum(tmp_path):
    img_path = tmp_path / "plot.png"
    y_true = _render_test_plot(
        img_path,
        {"Carbonyl (Ketone / Aldehyde / Ester / Carboxylic Acid)", "Alkane"},
        seed=7,
    )

    spectrum = digitize_plot(str(img_path), x_range=(4000, 400), y_range=(0, 1.05))
    y_recovered = np.interp(GRID[::-1], spectrum.wavenumbers[::-1], spectrum.intensities[::-1])[::-1]

    mean_err = np.abs(y_recovered - y_true).mean()
    assert mean_err < 0.02  # tight overall fit; sharp peak tips may still be individually rounded


def test_digitized_spectrum_classifies_correctly(tmp_path):
    img_path = tmp_path / "plot.png"
    _render_test_plot(
        img_path,
        {"Carbonyl (Ketone / Aldehyde / Ester / Carboxylic Acid)", "Alkane"},
        seed=7,
    )

    spectrum = digitize_plot(str(img_path), x_range=(4000, 400), y_range=(0, 1.05))
    spectrum.intensities = correct_baseline(spectrum.intensities)
    spectrum.intensities = smooth(spectrum.intensities)
    spectrum = normalize(spectrum)
    peaks = detect_peaks(spectrum, height=0.05, prominence=0.03, distance=15)
    predictions = classify(peaks)

    top_group = predictions[0].functional_group
    assert top_group == "Carbonyl (Ketone / Aldehyde / Ester / Carboxylic Acid)"


def test_calibration_profile_roundtrip(tmp_path):
    img_path = tmp_path / "plot.png"
    _render_test_plot(img_path, {"Alcohol / Phenol"}, seed=3)
    profile_path = tmp_path / "profile.json"

    profile = calibrate_and_save(str(img_path), x_range=(4000, 400), y_range=(0, 1.05),
                                  profile_path=str(profile_path))
    assert profile_path.exists()

    spectrum = digitize_plot(str(img_path), profile_path=str(profile_path))
    assert len(spectrum.wavenumbers) > 100
    assert spectrum.wavenumbers[0] > spectrum.wavenumbers[-1]  # descending, per convention


def test_digitize_raises_without_axis_range_or_profile(tmp_path):
    img_path = tmp_path / "plot.png"
    _render_test_plot(img_path, {"Alkane"}, seed=2)
    with pytest.raises(ValueError):
        digitize_plot(str(img_path))
