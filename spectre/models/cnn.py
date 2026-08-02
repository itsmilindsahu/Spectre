"""
cnn.py

Phase 3 model: a 1D-CNN functional-group classifier, trained on raw
(preprocessed but not peak-picked) IR spectra. Unlike the rule engine, this
model doesn't need explicit peak detection -- it learns directly from the
shape of the curve, which is the point: it should hold up better on noisy,
overlapping, real-world spectra where individual peaks are hard to isolate
(that's the Phase 3 goal in the roadmap).

Output format matches `models.rule_engine.Prediction` so the CLI and any
downstream tooling (e.g. the web tool) can treat rule-engine and CNN
predictions identically -- just no `evidence`/`matched_rules` peaks, since
the CNN doesn't reason peak-by-peak (see `notes` for a hint about that
tradeoff instead).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "spectre.models.cnn requires torch. Install it with:\n"
        "    pip install torch\n"
        "(it's commented out in requirements.txt until Phase 3 is the "
        "active track, to keep the Phase 1 rule-engine install lightweight)"
    ) from e

from spectre.ingestion.parsers import Spectrum
from spectre.preprocessing.baseline import correct_baseline
from spectre.preprocessing.grid import resample, normalize
from spectre.preprocessing.smoothing import smooth
from spectre.synthetic.generator import GRID, LABELS, N_POINTS


@dataclass
class CNNPrediction:
    functional_group: str
    confidence: float
    evidence: List = field(default_factory=list)   # always empty -- CNN has no peak-level evidence
    matched_rules: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "functional_group": self.functional_group,
            "confidence": round(self.confidence, 3),
            "evidence_peaks_cm-1": [],
            "notes": ["CNN prediction (Phase 3) -- no peak-level evidence, "
                      "cross-check against --model rule for interpretability"],
        }


class SpectreCNN(nn.Module):
    """
    Small 1D-CNN over a fixed-length (N_POINTS,) spectrum vector.
    Multi-label output (sigmoid per class, not softmax) since a spectrum can
    show several functional groups at once.
    """

    def __init__(self, n_labels: int, n_points: int = N_POINTS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=9, padding=4), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_labels),
        )

    def forward(self, x):
        # x: (batch, n_points) -> (batch, 1, n_points)
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.head(x)  # raw logits -- apply sigmoid outside for probabilities


def save_checkpoint(model: "SpectreCNN", labels: List[str], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": labels}, path)


def load_checkpoint(path: str, device: str = "cpu") -> tuple["SpectreCNN", List[str]]:
    ckpt = torch.load(path, map_location=device)
    labels = ckpt["labels"]
    model = SpectreCNN(n_labels=len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, labels


def _preprocess_for_cnn(spectrum: Spectrum) -> np.ndarray:
    """Same preprocessing as the rule-engine pipeline, then resample onto the
    fixed grid the CNN was trained on."""
    s = Spectrum(
        wavenumbers=spectrum.wavenumbers.copy(),
        intensities=correct_baseline(spectrum.intensities),
        metadata=spectrum.metadata,
    )
    s.intensities = smooth(s.intensities)
    s = normalize(s)
    s = resample(s, grid=GRID)
    s = normalize(s)  # renormalize post-resample in case interpolation introduced edge zeros
    return s.intensities.astype(np.float32)


def predict(spectrum: Spectrum, model: "SpectreCNN", labels: List[str],
            threshold: float = 0.5) -> List[CNNPrediction]:
    """Run the CNN on a spectrum and return CNNPrediction objects, sorted by
    confidence descending, thresholded at `threshold` (like the rule engine,
    only groups above threshold are returned -- pass threshold=0.0 to see all)."""
    x = _preprocess_for_cnn(spectrum)
    with torch.no_grad():
        logits = model(torch.from_numpy(x).unsqueeze(0))
        probs = torch.sigmoid(logits).squeeze(0).numpy()

    preds = [
        CNNPrediction(functional_group=label, confidence=float(p))
        for label, p in zip(labels, probs)
        if p >= threshold
    ]
    preds.sort(key=lambda p: p.confidence, reverse=True)
    return preds
