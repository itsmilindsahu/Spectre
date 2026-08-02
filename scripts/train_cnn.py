"""
train_cnn.py

Trains the Phase 3 1D-CNN (`spectre.models.cnn.SpectreCNN`) on synthetic
spectra generated from the correlation table (`spectre.synthetic.generator`)
since a real labeled dataset (M3 in the roadmap) doesn't exist yet.

Usage:
    python scripts/train_cnn.py
    python scripts/train_cnn.py --n-train 20000 --epochs 15 --out models_ckpt/cnn_ir.pt

Swap-in path for real data later: once `data/labeled/` has real
SMARTS-labeled spectra, replace `generate_dataset()` below with a loader
over that directory -- `SpectreCNN`, the training loop, and the eval report
all stay the same, since they only depend on (X, Y, labels) arrays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import TensorDataset, DataLoader

from spectre.models.cnn import SpectreCNN, save_checkpoint
from spectre.synthetic.generator import generate_dataset, LABELS


def train(n_train: int, n_val: int, epochs: int, batch_size: int, lr: float,
          out_path: str, seed: int = 0):
    print(f"Generating {n_train} synthetic training spectra + {n_val} validation spectra...")
    X_train, Y_train, labels = generate_dataset(n_train, seed=seed)
    X_val, Y_val, _ = generate_dataset(n_val, seed=seed + 1)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = SpectreCNN(n_labels=len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"Training on {len(labels)} labels: {labels}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                all_preds.append(torch.sigmoid(logits).numpy())
                all_targets.append(yb.numpy())
        val_loss /= len(val_ds)
        preds = np.concatenate(all_preds) >= 0.5
        targets = np.concatenate(all_targets) >= 0.5
        exact_match = (preds == targets).all(axis=1).mean()

        print(f"  epoch {epoch:>3}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_exact_match={exact_match:.3f}")

    save_checkpoint(model, labels, out_path)
    print(f"\nSaved checkpoint to {out_path}")
    _per_label_report(model, X_val, Y_val, labels)


def _per_label_report(model, X_val, Y_val, labels):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X_val))).numpy()
    preds = probs >= 0.5

    print("\nPer-label validation report (synthetic held-out set):")
    print("-" * 72)
    print(f"{'functional group':<45} {'precision':>9} {'recall':>8} {'support':>8}")
    for i, label in enumerate(labels):
        tp = np.sum((preds[:, i] == 1) & (Y_val[:, i] == 1))
        fp = np.sum((preds[:, i] == 1) & (Y_val[:, i] == 0))
        fn = np.sum((preds[:, i] == 0) & (Y_val[:, i] == 1))
        support = np.sum(Y_val[:, i] == 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"{label:<45} {precision:>9.3f} {recall:>8.3f} {support:>8.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Spectre Phase 3 CNN on synthetic spectra.")
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-val", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="models_ckpt/cnn_ir.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(args.n_train, args.n_val, args.epochs, args.batch_size, args.lr, args.out, args.seed)
