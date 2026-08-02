"""Tests for the Phase 3 CNN model and its synthetic-data bootstrap."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spectre.ingestion.parsers import Spectrum
from spectre.models.cnn import SpectreCNN, save_checkpoint, load_checkpoint, predict
from spectre.synthetic.generator import generate_dataset, generate_spectrum, LABELS, GRID, N_POINTS


def test_generate_dataset_shapes():
    X, Y, labels = generate_dataset(n_samples=10, seed=1)
    assert X.shape == (10, N_POINTS)
    assert Y.shape == (10, len(LABELS))
    assert labels == LABELS
    # every row should be normalized to [0, 1]
    assert X.min() >= 0.0
    assert X.max() <= 1.0 + 1e-6


def test_generate_spectrum_reflects_requested_groups():
    rng = np.random.default_rng(42)
    # dropout_prob=0 and decoy_prob=0 so the requested peak is guaranteed present
    y = generate_spectrum(rng, active_groups={"Alkane"}, dropout_prob=0.0, decoy_prob=0.0, noise_level=0.0)
    # Alkane's rule range is 2950-2850 cm^-1; there should be a real bump there
    mask = (GRID <= 2950) & (GRID >= 2850)
    assert y[mask].max() > 0.3


def test_cnn_forward_pass_shape():
    model = SpectreCNN(n_labels=len(LABELS))
    x = torch.zeros(4, N_POINTS)
    out = model(x)
    assert out.shape == (4, len(LABELS))


def test_checkpoint_roundtrip_and_predict(tmp_path):
    model = SpectreCNN(n_labels=len(LABELS))
    ckpt_path = tmp_path / "cnn_ir.pt"
    save_checkpoint(model, LABELS, str(ckpt_path))

    loaded_model, labels = load_checkpoint(str(ckpt_path))
    assert labels == LABELS

    spectrum = Spectrum(
        wavenumbers=GRID.copy(),
        intensities=np.random.default_rng(0).random(N_POINTS).astype(np.float32),
    )
    preds = predict(spectrum, loaded_model, labels, threshold=0.0)
    assert len(preds) == len(LABELS)
    for p in preds:
        assert 0.0 <= p.confidence <= 1.0
        assert p.evidence == []
