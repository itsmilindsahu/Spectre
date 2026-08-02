"""
cli.py

Spectre command-line tool: input a spectrum file, get functional-group
predictions out. This is the foremost interface for the project right now --
a UI can be layered on top of this same pipeline later without changing
any of the underlying stages.

Usage:
    python -m spectre.cli path/to/spectrum.csv
    python -m spectre.cli path/to/spectrum.jdx --plot
    python -m spectre.cli path/to/spectrum.csv --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spectre.ingestion.parsers import load_spectrum
from spectre.preprocessing.baseline import correct_baseline
from spectre.preprocessing.grid import normalize
from spectre.preprocessing.smoothing import smooth
from spectre.features.peaks import detect_peaks
from spectre.models.rule_engine import classify


def run_pipeline(path: str, height: float = 0.05, prominence: float = 0.03,
                  distance: int = 15, model: str = "rule", checkpoint: str = "models_ckpt/cnn_ir.pt",
                  x_min: float | None = None, x_max: float | None = None,
                  y_min: float | None = None, y_max: float | None = None,
                  calibration: str | None = None):
    """Run ingestion -> preprocessing -> features -> classifier, end to end.

    model="rule": Phase 1 correlation-table rule engine (default, always available).
    model="cnn":  Phase 3 1D-CNN (requires torch + a trained checkpoint --
                  see scripts/train_cnn.py -- and skips peak-level evidence).

    If `path` is a .png/.jpg/.jpeg, it's treated as a plotted spectrum image
    and digitized first (see spectre/ingestion/image_digitizer.py) -- pass
    either --calibration (a saved profile from a previous run) or all of
    --x-min/--x-max/--y-min/--y-max (the real axis range that image's plot
    uses).
    """
    if Path(path).suffix.lower() in (".png", ".jpg", ".jpeg"):
        from spectre.ingestion.image_digitizer import digitize_plot
        if calibration:
            spectrum = digitize_plot(path, profile_path=calibration)
        else:
            if None in (x_min, x_max, y_min, y_max):
                raise ValueError(
                    "Digitizing a plot image requires --x-min/--x-max/--y-min/--y-max "
                    "(the real axis range of that plot) or --calibration <saved profile>."
                )
            spectrum = digitize_plot(path, x_range=(x_min, x_max), y_range=(y_min, y_max))
    else:
        spectrum = load_spectrum(path)

    spectrum.intensities = correct_baseline(spectrum.intensities)
    spectrum.intensities = smooth(spectrum.intensities)
    spectrum = normalize(spectrum)

    peaks = detect_peaks(spectrum, height=height, prominence=prominence, distance=distance)

    if model == "cnn":
        from spectre.models.cnn import load_checkpoint, predict as cnn_predict
        cnn_model, labels = load_checkpoint(checkpoint)
        predictions = cnn_predict(spectrum, cnn_model, labels)
    else:
        predictions = classify(peaks)

    return spectrum, peaks, predictions


def _print_report(path: str, peaks, predictions):
    print(f"\nSpectre report — {path}")
    print("=" * 60)
    print(f"Detected peaks: {len(peaks)}")
    for p in peaks:
        print(f"  {p.wavenumber:7.1f} cm^-1   intensity={p.intensity:.2f}   "
              f"width={p.width:.1f}   shape={p.shape}")

    print("\nPredicted functional groups:")
    print("-" * 60)
    if not predictions:
        print("  (none confidently matched -- try lowering --height/--prominence)")
    for pred in predictions:
        print(f"  {pred.functional_group:<45} confidence={pred.confidence:.2f}")
        for note in pred.matched_rules:
            print(f"      - {note}")
    print()


def _plot(spectrum, peaks, path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping plot (pip install matplotlib)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spectrum.wavenumbers, spectrum.intensities, linewidth=1)
    for p in peaks:
        ax.axvline(p.wavenumber, color="red", linestyle="--", alpha=0.3)
        ax.annotate(f"{p.wavenumber:.0f}", (p.wavenumber, p.intensity),
                    fontsize=7, rotation=90, va="bottom")
    ax.invert_xaxis()  # IR convention: high wavenumber on the left
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Normalized absorbance")
    ax.set_title(f"Spectre — {path}")
    out_path = "spectre_plot.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Spectre: identify functional groups from an IR spectrum."
    )
    parser.add_argument("spectrum_file", help="Path to a .csv, .tsv, .jdx spectrum file, or a .png/.jpg plot image")
    parser.add_argument("--height", type=float, default=0.05,
                         help="Minimum normalized peak height (0-1) to count as a peak")
    parser.add_argument("--prominence", type=float, default=0.03,
                         help="Minimum peak prominence (0-1)")
    parser.add_argument("--distance", type=int, default=15,
                         help="Minimum distance between peaks, in samples")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--plot", action="store_true", help="Save an annotated spectrum plot")
    parser.add_argument("--model", choices=["rule", "cnn"], default="rule",
                         help="Classifier to use: 'rule' (Phase 1, default) or 'cnn' (Phase 3, requires torch + a trained checkpoint)")
    parser.add_argument("--checkpoint", type=str, default="models_ckpt/cnn_ir.pt",
                         help="Path to a CNN checkpoint (only used with --model cnn)")
    parser.add_argument("--x-min", type=float, default=None, help="Real axis x-value at the left edge of a plot image (image input only)")
    parser.add_argument("--x-max", type=float, default=None, help="Real axis x-value at the right edge of a plot image (image input only)")
    parser.add_argument("--y-min", type=float, default=None, help="Real axis y-value at the bottom edge of a plot image (image input only)")
    parser.add_argument("--y-max", type=float, default=None, help="Real axis y-value at the top edge of a plot image (image input only)")
    parser.add_argument("--calibration", type=str, default=None,
                         help="Path to a saved calibration profile (see spectre.ingestion.image_digitizer) -- reuse instead of passing --x-min etc. every time")
    args = parser.parse_args()

    try:
        spectrum, peaks, predictions = run_pipeline(
            args.spectrum_file, height=args.height, prominence=args.prominence,
            distance=args.distance, model=args.model, checkpoint=args.checkpoint,
            x_min=args.x_min, x_max=args.x_max, y_min=args.y_min, y_max=args.y_max,
            calibration=args.calibration,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        result = {
            "source_file": args.spectrum_file,
            "peaks": [
                {"wavenumber_cm-1": round(p.wavenumber, 1),
                 "intensity": round(p.intensity, 3),
                 "width_cm-1": round(p.width, 1),
                 "shape": p.shape}
                for p in peaks
            ],
            "predictions": [p.to_dict() for p in predictions],
        }
        print(json.dumps(result, indent=2))
    else:
        _print_report(args.spectrum_file, peaks, predictions)

    if args.plot:
        _plot(spectrum, peaks, args.spectrum_file)


if __name__ == "__main__":
    main()
