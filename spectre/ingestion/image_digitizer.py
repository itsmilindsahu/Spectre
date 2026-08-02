"""
image_digitizer.py

Lets Spectre accept a PNG (or JPG) of a *plotted* spectrum -- e.g. an export
from one consistent instrument software -- instead of requiring the raw
(wavenumber, absorbance) numbers.

This is NOT a general "read any spectrum screenshot" tool. It assumes:
  1. The plot has a visible rectangular axis frame (the standard matplotlib/
     Origin/instrument-software look: a box with tick marks on all sides).
  2. There is exactly one curve, drawn in a single, roughly-consistent color,
     that is a single-valued function of x (one y per x -- true for IR/UV-Vis/
     NMR line plots, not for e.g. scatter clouds).
  3. You already know (once, per source) what real-world (x_min, x_max,
     y_min, y_max) the plot axes correspond to -- these can't be recovered
     from pixels alone without OCR-ing tick labels, which is far less
     reliable than just telling the digitizer once and reusing a saved
     calibration profile for every future export from the same software.

Workflow:
    profile = calibrate_and_save(
        "sample_export.png", x_range=(4000, 400), y_range=(0.0, 1.2),
        profile_path="calibration/my_instrument.json",
    )
    spectrum = digitize_plot("new_export.png", profile_path="calibration/my_instrument.json")

Or one-shot without saving a profile:
    spectrum = digitize_plot("export.png", x_range=(4000, 400), y_range=(0.0, 1.2))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from spectre.ingestion.parsers import Spectrum


@dataclass
class CalibrationProfile:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    bbox: Tuple[int, int, int, int]        # (left, top, right, bottom) in pixels
    curve_color: Tuple[int, int, int]       # RGB of the traced line
    color_tolerance: int = 40

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CalibrationProfile":
        with open(path) as f:
            data = json.load(f)
        data["bbox"] = tuple(data["bbox"])
        data["curve_color"] = tuple(data["curve_color"])
        return cls(**data)


def _load_rgb(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    return np.array(img)  # (H, W, 3)


def detect_plot_bbox(rgb: np.ndarray, min_line_frac: float = 0.5) -> Tuple[int, int, int, int]:
    """
    Find the axis frame by looking for the longest near-continuous dark
    horizontal and vertical lines -- the standard boxed-axes look. Returns
    (left, top, right, bottom) in pixel coordinates.
    """
    h, w, _ = rgb.shape
    gray = rgb.mean(axis=2)
    dark = gray < 128  # axis lines/ticks are near-black

    row_frac = dark.mean(axis=1)  # fraction of dark pixels per row
    col_frac = dark.mean(axis=0)  # fraction of dark pixels per column

    row_candidates = np.where(row_frac > min_line_frac)[0]
    col_candidates = np.where(col_frac > min_line_frac)[0]

    if len(row_candidates) < 2 or len(col_candidates) < 2:
        raise ValueError(
            "Could not detect a rectangular axis frame (no rows/columns dark "
            "enough to be axis spines). This digitizer needs a plot with a "
            "visible boxed frame -- try passing bbox= manually if your plots "
            "don't have one."
        )

    top, bottom = int(row_candidates.min()), int(row_candidates.max())
    left, right = int(col_candidates.min()), int(col_candidates.max())
    return (left, top, right, bottom)


def detect_curve_color(rgb: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    """
    Find the dominant non-black/white/gray color inside the plot area --
    that's almost always the traced curve, since axis frame/gridlines/text
    are black/gray and the background is white.
    """
    left, top, right, bottom = bbox
    region = rgb[top:bottom, left:right].reshape(-1, 3).astype(int)

    is_grayish = (region.max(axis=1) - region.min(axis=1)) < 20  # R≈G≈B
    is_near_white = region.min(axis=1) > 230
    colorful = region[~is_grayish & ~is_near_white]

    if len(colorful) > 200:
        colors, counts = np.unique(colorful, axis=0, return_counts=True)
        return tuple(int(c) for c in colors[np.argmax(counts)])

    # Fallback: monochrome plot (e.g. plain black line) -- use dark, non-axis-frame pixels.
    is_dark = region.max(axis=1) < 100
    dark_pixels = region[is_dark]
    if len(dark_pixels) == 0:
        raise ValueError("Could not find a distinguishable curve color inside the plot area.")
    colors, counts = np.unique(dark_pixels, axis=0, return_counts=True)
    return tuple(int(c) for c in colors[np.argmax(counts)])


def _extract_curve_pixels(rgb: np.ndarray, bbox: Tuple[int, int, int, int],
                           curve_color: Tuple[int, int, int], tolerance: int) -> np.ndarray:
    """
    For each pixel-column inside bbox, find matching-color pixels and take
    their mean row -- assumes the curve is single-valued in x. Returns an
    array of length (right-left) with the row (float, may be NaN where no
    match was found in that column).
    """
    left, top, right, bottom = bbox
    region = rgb[top:bottom, left:right].astype(int)
    color = np.array(curve_color)

    dist = np.sqrt(((region - color) ** 2).sum(axis=2))
    match = dist <= tolerance  # (rows, cols) boolean

    n_cols = match.shape[1]
    rows = np.arange(match.shape[0])[:, None]
    row_of_col = np.full(n_cols, np.nan)
    for c in range(n_cols):
        matched_rows = rows[match[:, c]]
        if len(matched_rows) > 0:
            row_of_col[c] = matched_rows.mean()
    return row_of_col


def _pixels_to_spectrum(row_of_col: np.ndarray, bbox: Tuple[int, int, int, int],
                         x_range: Tuple[float, float], y_range: Tuple[float, float],
                         source: str) -> Spectrum:
    left, top, right, bottom = bbox
    n_cols = right - left
    x_min, x_max = x_range
    y_min, y_max = y_range

    valid = ~np.isnan(row_of_col)
    if valid.sum() < n_cols * 0.5:
        raise ValueError(
            f"Only recovered the curve in {valid.sum()}/{n_cols} columns -- "
            "curve_color/tolerance is probably wrong for this image."
        )

    col_idx = np.arange(n_cols)
    # Fill small gaps by linear interpolation over columns where the curve
    # color wasn't matched (anti-aliased edges, gridline occlusion, etc.)
    row_filled = np.interp(col_idx, col_idx[valid], row_of_col[valid])

    # Column pixel -> x value (linear across the bbox width)
    wavenumbers = x_min + (col_idx / (n_cols - 1)) * (x_max - x_min)
    # Row pixel -> y value: row 0 is the TOP of the plot = y_max, row (bbox height) = y_min
    bbox_h = bottom - top
    intensities = y_max - (row_filled / (bbox_h - 1)) * (y_max - y_min)

    return Spectrum(
        wavenumbers=wavenumbers.astype(float),
        intensities=intensities.astype(float),
        metadata={"source": source, "digitized_from_image": True},
    )


def digitize_plot(image_path: str,
                   x_range: Optional[Tuple[float, float]] = None,
                   y_range: Optional[Tuple[float, float]] = None,
                   bbox: Optional[Tuple[int, int, int, int]] = None,
                   curve_color: Optional[Tuple[int, int, int]] = None,
                   color_tolerance: int = 40,
                   profile_path: Optional[str] = None) -> Spectrum:
    """
    Reconstruct a Spectrum from a plot image.

    Either pass x_range/y_range (+ optionally bbox/curve_color to skip
    auto-detection), or pass profile_path to reuse a saved
    CalibrationProfile (see calibrate_and_save below).
    """
    if profile_path is not None:
        profile = CalibrationProfile.load(profile_path)
        x_range = (profile.x_min, profile.x_max)
        y_range = (profile.y_min, profile.y_max)
        bbox = profile.bbox
        curve_color = profile.curve_color
        color_tolerance = profile.color_tolerance

    if x_range is None or y_range is None:
        raise ValueError("Must supply x_range and y_range (or a profile_path).")

    rgb = _load_rgb(image_path)
    if bbox is None:
        bbox = detect_plot_bbox(rgb)
    if curve_color is None:
        curve_color = detect_curve_color(rgb, bbox)

    row_of_col = _extract_curve_pixels(rgb, bbox, curve_color, color_tolerance)
    spectrum = _pixels_to_spectrum(row_of_col, bbox, x_range, y_range, source=image_path)

    # Spectrum enforces descending wavenumber order internally; if x_range
    # was given ascending (x_min < x_max) the array above is ascending too,
    # so flip if needed to match convention (Spectrum.__post_init__ actually
    # handles this automatically -- see ingestion/parsers.py).
    return spectrum


def calibrate_and_save(image_path: str, x_range: Tuple[float, float], y_range: Tuple[float, float],
                        profile_path: str, bbox: Optional[Tuple[int, int, int, int]] = None,
                        curve_color: Optional[Tuple[int, int, int]] = None,
                        color_tolerance: int = 40) -> CalibrationProfile:
    """
    Run auto-detection once on a representative image from a given source,
    inspect/confirm the result, then save it so every future export from
    that same instrument software can be digitized without re-detecting.
    """
    rgb = _load_rgb(image_path)
    if bbox is None:
        bbox = detect_plot_bbox(rgb)
    if curve_color is None:
        curve_color = detect_curve_color(rgb, bbox)

    profile = CalibrationProfile(
        x_min=x_range[0], x_max=x_range[1],
        y_min=y_range[0], y_max=y_range[1],
        bbox=bbox, curve_color=curve_color, color_tolerance=color_tolerance,
    )
    profile.save(profile_path)
    return profile
