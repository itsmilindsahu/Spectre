"""
generator.py

Phase 3 needs labeled training data. M3 (a 500+ compound SMARTS-labeled
dataset from real spectra) hasn't happened yet -- `data/labeled/` is still
empty. Rather than block the CNN on that, this module synthesizes spectra
directly from the same domain knowledge already encoded in
`models/correlation_table.py`: for each functional group we know its
diagnostic wavenumber region and expected peak shape, so we can place
randomized Gaussian peaks inside those regions, add decoys/noise/baseline
drift, and get a multi-label (spectrum, functional_groups_present) dataset
of arbitrary size for free.

This is a bootstrap, not a replacement for real data -- swap in
`data/labeled/` once M3 lands and the CNN will train the same way (see
`scripts/train_cnn.py`).
"""

from __future__ import annotations

import numpy as np

from spectre.models.correlation_table import CORRELATION_TABLE

# Unique functional-group labels, in a fixed, stable order (this order is
# saved alongside every checkpoint so inference always maps indices back to
# the right label).
LABELS = sorted({rule.group for rule in CORRELATION_TABLE})

GRID = np.arange(4000, 399, -1.0)  # matches preprocessing/grid.py DEFAULT_GRID
N_POINTS = len(GRID)


def _gaussian_peak(x: np.ndarray, center: float, height: float, width_cm1: float) -> np.ndarray:
    """width_cm1 is treated as FWHM; convert to Gaussian sigma."""
    sigma = max(width_cm1, 4.0) / 2.3548
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _shape_to_width_range(shape: str) -> tuple[float, float]:
    if shape == "broad":
        return (150.0, 400.0)
    if shape == "medium":
        return (50.0, 150.0)
    if shape == "sharp":
        return (10.0, 45.0)
    return (10.0, 400.0)  # "variable"


def generate_spectrum(rng: np.random.Generator, active_groups: set[str],
                       noise_level: float = 0.02, dropout_prob: float = 0.15,
                       decoy_prob: float = 0.4) -> np.ndarray:
    """
    Build one synthetic spectrum (on GRID) whose active peaks correspond to
    `active_groups`. Returns a 1D float32 array, normalized to [0, 1].

    Realism knobs, deliberately imperfect (real spectra are messy):
      - dropout_prob: chance a group's expected peak is simply missing
        (weak/overlapping/instrument noise in practice).
      - decoy_prob: chance of adding an unrelated, non-labeled peak
        (forces the model to learn shape+position jointly, not "any peak here").
      - noise_level: additive Gaussian noise, plus a slow baseline wobble.
    """
    y = np.zeros_like(GRID)

    rules_by_group: dict[str, list] = {}
    for rule in CORRELATION_TABLE:
        rules_by_group.setdefault(rule.group, []).append(rule)

    for group in active_groups:
        if rng.random() < dropout_prob:
            continue
        rule = rules_by_group[group][rng.integers(0, len(rules_by_group[group]))]
        center = rng.uniform(rule.high, rule.low)
        lo_w, hi_w = _shape_to_width_range(rule.shape)
        width = rng.uniform(lo_w, hi_w)
        height = rng.uniform(0.5, 1.0)
        y += _gaussian_peak(GRID, center, height, width)

    # Unlabeled decoy peaks -- e.g. every organic molecule has *some* C-H
    # stretch, and real spectra have small shoulders that mean nothing.
    n_decoys = rng.poisson(1.5) if rng.random() < decoy_prob else 0
    for _ in range(n_decoys):
        center = rng.uniform(450, 3950)
        width = rng.uniform(15, 120)
        height = rng.uniform(0.05, 0.35)
        y += _gaussian_peak(GRID, center, height, width)

    # Slow baseline wobble (low-frequency sine mix) -- correct_baseline()
    # should remove most of this, but training on the raw+noisy signal too
    # makes the CNN robust even if a user skips preprocessing.
    n = len(GRID)
    t = np.linspace(0, 1, n)
    wobble = (
        0.03 * rng.uniform(-1, 1) * np.sin(2 * np.pi * rng.uniform(0.5, 2) * t)
        + 0.02 * rng.uniform(-1, 1) * t
    )
    y += wobble

    y += rng.normal(0, noise_level, size=n)
    y = np.clip(y, 0, None)

    y_max = y.max() if y.max() > 1e-6 else 1.0
    return (y / y_max).astype(np.float32)


def generate_dataset(n_samples: int, seed: int = 0, max_groups: int = 4):
    """
    Returns (X, Y, labels):
      X: float32 array (n_samples, N_POINTS)
      Y: float32 multi-hot array (n_samples, len(LABELS))
      labels: LABELS (fixed order, for index -> group_name lookup)
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, N_POINTS), dtype=np.float32)
    Y = np.zeros((n_samples, len(LABELS)), dtype=np.float32)

    for i in range(n_samples):
        k = rng.integers(1, max_groups + 1)
        active = set(rng.choice(LABELS, size=k, replace=False))
        X[i] = generate_spectrum(rng, active)
        for group in active:
            Y[i, LABELS.index(group)] = 1.0

    return X, Y, LABELS
