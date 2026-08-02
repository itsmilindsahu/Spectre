# Spectre
### An Open-Source Pipeline for Automated Spectral Interpretation (IR → NMR)

**Authors:** [Your Name] · Paarth
**Status:** Proposal / Pre-alpha
**License:** MIT (proposed)

---

## 0. Quickstart (working code — Phase 1 is implemented)

```bash
git clone <this-repo>
cd spectre
pip install -r requirements.txt

# Run on the included synthetic example spectra
python -m spectre.cli examples/sample_spectrum.csv          # carboxylic-acid-like
python -m spectre.cli examples/sample_spectrum_alcohol.csv  # alcohol-like

# Machine-readable output
python -m spectre.cli examples/sample_spectrum.csv --json

# Save an annotated plot
python -m spectre.cli examples/sample_spectrum.csv --plot

# Run the test suite
pytest tests/ -v
```

The rule-based classifier (Phase 1 from the roadmap below) is fully
implemented and tested: ingestion (CSV + minimal JCAMP-DX), baseline
correction, smoothing, peak detection, and the correlation-table rule
engine all run end to end. Phases 2–5 (ML, deep learning, NMR) are
architected for but not yet built — see the roadmap.

---

## 1. Abstract

Spectroscopic databases (NIST WebBook, SDBS, Coblentz) let a chemist look up the spectrum of a compound they already know. None of them solve the inverse problem: **given a raw, unlabeled spectrum, automatically determine what functional groups — and eventually what structure — produced it.**

**Spectre** is an open-source pipeline that starts by solving this for Infrared (IR) spectroscopy — classifying functional groups directly from spectral data — and is architected from day one to extend to Nuclear Magnetic Resonance (NMR) spectroscopy as a second modality, with an eventual goal of multi-modal structure elucidation (IR + NMR + MS, the way a chemist actually works).

The project is split along two tracks that mirror the authors' backgrounds:
- **Computational track** ([Your Name]): pipeline architecture, data engineering, ML/DL modeling, software engineering.
- **Chemistry track** (Paarth): spectral interpretation rules, structure–label generation (SMARTS pattern design), domain validation, dataset curation.

---

## 2. Motivation

Functional-group identification from a spectrum is currently done by:
1. **Manual interpretation** — a trained chemist scans peak positions, shapes, and intensities against memorized correlation tables (see `docs/correlation-table.md`, based on our original study notes).
2. **Database lookup** — comparing a spectrum against a reference library, which only works if the exact compound is already catalogued.

Neither approach scales to novel or unknown compounds, and neither is automatable at scale for large screening workflows (e.g. environmental monitoring, quality control, high-throughput chemical screening, or teaching tools for students still learning to read spectra).

Spectre aims to close this gap with a transparent, reproducible, and freely available tool — built openly so the broader chemistry and cheminformatics community can inspect, correct, and extend it.

---

## 3. Goals

| Phase | Goal | Output |
|---|---|---|
| 1 | Rule-based IR functional-group classifier | Baseline accuracy benchmark, interpretable peak-to-label mapping |
| 2 | ML-based IR classifier trained on labeled spectral data | Multi-label functional group predictor, outperforming rule baseline |
| 3 | Deep learning model (1D-CNN) on raw IR spectra | Higher accuracy, handles overlapping/noisy real-world spectra |
| 4 | NMR module (¹H / ¹³C) using the same pipeline architecture | Chemical shift → environment/functional-group classifier |
| 5 | Multi-modal fusion (IR + NMR) | Combined structure elucidation assistant |

---

## 4. Pipeline Architecture

```
                        ┌─────────────────────────┐
                        │   Raw Spectral Input     │
                        │  (JCAMP-DX / CSV / etc.) │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │   1. Ingestion Layer      │
                        │   - Format parsing        │
                        │   - Metadata extraction   │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │   2. Preprocessing        │
                        │   - Baseline correction   │
                        │   - Grid interpolation    │
                        │   - Noise smoothing       │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │   3. Feature Extraction   │
                        │   - Peak picking          │
                        │   - Peak shape/width       │
                        │   - Region binning        │
                        └────────────┬─────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
   ┌──────────▼─────────┐ ┌──────────▼─────────┐ ┌──────────▼─────────┐
   │ 4a. Rule Engine     │ │ 4b. Classical ML   │ │ 4c. Deep Learning   │
   │ (correlation table) │ │ (RF / GBM)         │ │ (1D-CNN)            │
   └──────────┬─────────┘ └──────────┬─────────┘ └──────────┬─────────┘
              └──────────────────────┼──────────────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │   5. Output Layer         │
                        │   - Functional group list │
                        │   - Confidence scores     │
                        │   - Peak-level evidence   │
                        └───────────────────────────┘
```

**Design principle:** every prediction must trace back to the peak(s) that caused it. A black-box label with no evidence is not acceptable output for a scientific tool — this is non-negotiable for both trust and debuggability.

---

## 5. Repository Structure (proposed)

```
spectre/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── .gitignore
├── docs/
│   ├── correlation-table.md        # IR peak-region reference (chemistry track)
│   └── nmr-extension.md            # Phase 4 design notes
├── data/
│   ├── raw/                        # Downloaded spectra (gitignored, empty by default)
│   ├── labeled/                    # Auto-labeled via SMARTS (gitignored, empty by default)
│   └── README.md                   # Data sourcing + licensing notes
├── examples/
│   ├── sample_spectrum.csv         # synthetic carboxylic-acid-like spectrum
│   └── sample_spectrum_alcohol.csv # synthetic alcohol-like spectrum
├── spectre/
│   ├── ingestion/
│   │   └── parsers.py              # CSV + minimal JCAMP-DX parser  [implemented]
│   ├── preprocessing/
│   │   ├── baseline.py             # ALS baseline correction        [implemented]
│   │   ├── smoothing.py            # Savitzky-Golay noise reduction [implemented]
│   │   └── grid.py                 # Resampling / normalization     [implemented]
│   ├── features/
│   │   └── peaks.py                # Peak picking + shape classification [implemented]
│   ├── models/
│   │   ├── correlation_table.py    # IR rule data                   [implemented]
│   │   ├── rule_engine.py          # Phase 1 classifier              [implemented]
│   │   ├── classical_ml.py         # Phase 2                         [planned]
│   │   └── cnn.py                  # Phase 3                         [planned]
│   ├── evaluation/                 # Benchmarking against known spectra [planned]
│   └── cli.py                      # End-to-end CLI entry point      [implemented]
└── tests/
    ├── test_parsers.py
    ├── test_peaks.py
    └── test_rule_engine.py
```

---

## 6. Data Strategy

1. **Source**: IR spectra pulled from NIST WebBook and SDBS (subject to each source's redistribution terms — raw data will not be re-hosted where licensing prohibits it; the pipeline will instead include fetch scripts).
2. **Format**: standardized on JCAMP-DX, parsed via the `jcamp` Python library, resampled onto a common wavenumber grid (4000–400 cm⁻¹).
3. **Auto-labeling**: for every compound with a known structure (SMILES), functional-group presence/absence labels are generated automatically using RDKit SMARTS pattern matching — this is the key mechanism that turns "known compound" into "labeled training example" without manual annotation, and is where the chemistry track (Paarth) contributes pattern definitions and validation.
4. **Held-out test set**: a manually curated set of spectra with chemist-verified labels, used only for final evaluation — never for training.

---

## 7. NMR Extension (Phase 4)

The same five-stage architecture is designed to generalize to NMR:

| IR concept | NMR equivalent |
|---|---|
| Wavenumber (cm⁻¹) axis | Chemical shift (ppm) axis |
| Peak shape (broad/sharp) | Peak multiplicity (splitting pattern) |
| Region correlation table | Chemical shift correlation table |
| Functional group from stretch | Proton/carbon environment from shift + coupling |

¹H NMR will be the first target (chemical shift + integration + splitting), followed by ¹³C. The rule engine, classical ML, and CNN layers are all format-agnostic by design — only the ingestion and feature-extraction layers need modality-specific implementations, which keeps the two tracks (IR now, NMR later) from requiring a rewrite.

---

## 8. Why Open Source

- **Reproducibility**: spectral interpretation tools used in research should be auditable, not black boxes.
- **Community correction**: functional-group correlation rules have known edge cases and exceptions; an open project lets domain experts submit corrections directly.
- **Education**: a transparent rule-engine layer doubles as a teaching tool for students learning to read spectra — the same audience this project started from.
- **Extensibility**: multi-modal spectral analysis (IR + NMR + MS) is a genuinely underserved open-source niche compared to the density of paid/proprietary tools.

---

## 9. Roadmap & Milestones

- [ ] **M1**: Ingestion + preprocessing pipeline working on real JCAMP-DX files
- [ ] **M2**: Rule-based classifier reproducing correlation-table logic, benchmarked on known compounds
- [ ] **M3**: Labeled dataset built via SMARTS auto-labeling (target: 500+ compounds)
- [ ] **M4**: Classical ML model beating rule-based baseline
- [ ] **M5**: 1D-CNN model, peak-evidence attribution for interpretability
- [ ] **M6**: Public release (PyPI package + documentation site)
- [ ] **M7**: NMR ingestion + feature extraction module
- [ ] **M8**: NMR classifier, feature parity with IR module

---

## 9.5 Web Tool

A self-contained, client-side front end (`web/index.html`) lets anyone in
the IISER network run a spectrum through the Phase 1 rule engine from a
browser, with no install — see `web/README.md` for deployment (GitHub
Pages) and central-logging setup.

## 10. Contributing

This project is in early proposal stage. Contribution guidelines, code of conduct, and issue templates will be added once the M1–M2 baseline is public. Early collaborators with cheminformatics, spectroscopy, or ML backgrounds are welcome to reach out.

---

## 11. Acknowledgements

Project originated from hand-written IR interpretation notes developed while studying spectral analysis fundamentals, later formalized into this pipeline.
