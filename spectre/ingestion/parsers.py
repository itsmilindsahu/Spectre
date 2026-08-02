"""
parsers.py

Ingestion layer: reads raw spectral files into a common in-memory format.

Common format used throughout the rest of the pipeline:
    Spectrum(
        wavenumbers: np.ndarray  # cm^-1, descending (4000 -> 400) by convention
        intensities: np.ndarray  # absorbance (higher = more absorbed)
        metadata: dict           # title, source, units, etc.
    )

Supported input formats:
    - CSV / TSV with two columns: wavenumber, intensity
    - JCAMP-DX (a common minimal subset: ##XYDATA=(X++(Y..Y)) tabular block,
      plus the common single-column-per-line variant). This is NOT a full
      JCAMP-DX spec implementation -- for production use, swap in the
      community `jcamp` PyPI package. This minimal parser exists so the
      pipeline is runnable without extra dependencies.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class Spectrum:
    wavenumbers: np.ndarray
    intensities: np.ndarray
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        # Enforce descending wavenumber order (IR convention: high energy, left)
        if len(self.wavenumbers) > 1 and self.wavenumbers[0] < self.wavenumbers[-1]:
            self.wavenumbers = self.wavenumbers[::-1]
            self.intensities = self.intensities[::-1]

    def __len__(self):
        return len(self.wavenumbers)


def load_spectrum(path: str | Path) -> Spectrum:
    """Auto-detect format from file extension and parse."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".txt"):
        return _parse_csv(path)
    if suffix in (".jdx", ".dx", ".jcamp"):
        return _parse_jcamp(path)
    raise ValueError(
        f"Unrecognized spectrum file extension '{suffix}'. "
        "Supported: .csv, .tsv, .txt, .jdx, .dx, .jcamp"
    )


def _parse_csv(path: Path) -> Spectrum:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    wavenumbers, intensities = [], []
    with open(path, newline="") as f:
        # Sniff for a header row
        sample = f.read(2048)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader, None)
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                x, y = float(row[0]), float(row[1])
            except ValueError:
                continue
            wavenumbers.append(x)
            intensities.append(y)

    if not wavenumbers:
        raise ValueError(f"No numeric data rows found in {path}")

    return Spectrum(
        wavenumbers=np.array(wavenumbers, dtype=float),
        intensities=np.array(intensities, dtype=float),
        metadata={"source_file": str(path), "format": "csv"},
    )


_HEADER_FIELD_RE = re.compile(r"^##(\w+)\s*=\s*(.*)$")


def _parse_jcamp(path: Path) -> Spectrum:
    """
    Minimal JCAMP-DX reader.

    Handles the common ##XYDATA=(X++(Y..Y)) block (X value followed by
    consecutive Y values on the same line, ASDF-free) as well as simple
    plain "x y" or "x, y" per line data blocks. This intentionally does not
    implement the full ASDF-compressed encoding used by some archives --
    for those files, use the `jcamp` PyPI package instead
    (`pip install jcamp`) and adapt this function accordingly.
    """
    header: Dict[str, str] = {}
    xy_lines = []
    in_data_block = False

    firstx = lastx = deltax = xfactor = yfactor = npoints = None

    with open(path, errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            m = _HEADER_FIELD_RE.match(line)
            if m:
                key, value = m.group(1).upper(), m.group(2).strip()
                header[key] = value
                if key == "FIRSTX":
                    firstx = float(value)
                elif key == "LASTX":
                    lastx = float(value)
                elif key == "DELTAX":
                    deltax = float(value)
                elif key == "XFACTOR":
                    xfactor = float(value)
                elif key == "YFACTOR":
                    yfactor = float(value)
                elif key == "NPOINTS":
                    npoints = int(float(value))
                elif key == "XYDATA":
                    in_data_block = True
                elif key == "END":
                    in_data_block = False
                continue

            if in_data_block:
                xy_lines.append(line)

    xfactor = xfactor or 1.0
    yfactor = yfactor or 1.0

    wavenumbers, intensities = [], []

    # Try "X++(Y Y Y ...)" style: first token is X, rest are consecutive Y's
    for line in xy_lines:
        tokens = re.split(r"[,\s]+", line.strip())
        tokens = [t for t in tokens if t not in ("", ",")]
        if len(tokens) < 2:
            continue
        try:
            nums = [float(t) for t in tokens]
        except ValueError:
            continue

        x0 = nums[0] * xfactor
        ys = [y * yfactor for y in nums[1:]]

        if deltax is not None:
            for i, y in enumerate(ys):
                wavenumbers.append(x0 + i * deltax)
                intensities.append(y)
        else:
            # Fall back to simple "x y" pairs
            if len(nums) == 2:
                wavenumbers.append(nums[0] * xfactor)
                intensities.append(nums[1] * yfactor)

    if not wavenumbers:
        raise ValueError(
            f"Could not extract XY data from {path}. This minimal JCAMP-DX "
            "parser does not support ASDF-compressed blocks -- try the "
            "`jcamp` PyPI package for full-spec files."
        )

    return Spectrum(
        wavenumbers=np.array(wavenumbers, dtype=float),
        intensities=np.array(intensities, dtype=float),
        metadata={"source_file": str(path), "format": "jcamp-dx", **header},
    )
