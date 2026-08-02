"""
export_onnx.py

Exports a trained SpectreCNN checkpoint (see scripts/train_cnn.py) to ONNX,
so the web tool (web/index.html) can run inference in-browser via
onnxruntime-web, with no Python backend.

Usage:
    python scripts/export_onnx.py --checkpoint models_ckpt/cnn_ir.pt --out web/cnn_ir.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from spectre.models.cnn import load_checkpoint
from spectre.synthetic.generator import N_POINTS


def export(checkpoint_path: str, out_path: str):
    model, labels = load_checkpoint(checkpoint_path)
    model.eval()

    dummy = torch.zeros(1, N_POINTS)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["spectrum"], output_names=["logits"],
        dynamic_axes={"spectrum": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    labels_path = str(Path(out_path).with_suffix("")) + "_labels.json"
    with open(labels_path, "w") as f:
        json.dump({"labels": labels, "n_points": N_POINTS}, f, indent=2)

    print(f"Exported ONNX model to {out_path}")
    print(f"Exported label list to {labels_path}")
    print(f"Labels ({len(labels)}): {labels}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a trained SpectreCNN checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, default="models_ckpt/cnn_ir.pt")
    parser.add_argument("--out", type=str, default="web/cnn_ir.onnx")
    args = parser.parse_args()
    export(args.checkpoint, args.out)
