"""
baseline.py

Baseline correction for raw spectra. Real instrument output rarely sits on a
flat zero baseline -- drift, scattering, and instrument artifacts tilt or
curve it. We use Asymmetric Least Squares (ALS) smoothing (Eilers, 2005),
a standard, well-tested baseline estimation method for spectroscopy.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01,
                  n_iter: int = 10) -> np.ndarray:
    """
    Estimate a baseline for signal `y` using Asymmetric Least Squares.

    Args:
        y: intensity array
        lam: smoothness parameter (larger = smoother baseline)
        p: asymmetry parameter (0 < p < 1; smaller = baseline hugs the
           bottom of the signal more, appropriate since absorbance peaks
           point up and the baseline should track the troughs between them)
        n_iter: number of reweighting iterations

    Returns:
        Estimated baseline array, same shape as y.
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L, format="csc")
    z = np.zeros(L)
    for _ in range(n_iter):
        W.setdiag(w)
        Z = (W + D).tocsc()
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def correct_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01) -> np.ndarray:
    """Subtract the estimated baseline from a raw intensity array."""
    baseline = als_baseline(y, lam=lam, p=p)
    corrected = y - baseline
    corrected[corrected < 0] = 0.0
    return corrected
