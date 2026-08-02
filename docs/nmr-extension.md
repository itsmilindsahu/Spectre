# NMR Extension — Design Notes (Phase 4)

Status: **planned, not yet implemented.** This document exists so the
architecture decisions are made deliberately up front, before any NMR code
is written, rather than bolted on later.

## Why the existing pipeline generalizes

Spectre's five-stage architecture (Ingestion → Preprocessing → Feature
Extraction → Model Layer → Output) was deliberately kept format-agnostic at
the interface level. Concretely, extending to NMR means:

| Stage | IR implementation (exists) | NMR implementation (to build) |
|---|---|---|
| Ingestion | `spectre/ingestion/parsers.py` (CSV, JCAMP-DX) | New parser for NMR file formats (e.g. JCAMP-DX also covers NMR; also common: `.fid`, `.dx`, vendor formats) |
| Preprocessing | Baseline correction, grid resampling (wavenumber) | Baseline/phase correction, chemical-shift referencing (ppm), possibly solvent-peak suppression |
| Feature Extraction | `spectre/features/peaks.py` (peak position, width, shape) | Peak position (ppm), multiplicity/splitting pattern, integration (relative peak area), coupling constants |
| Model Layer | `spectre/models/rule_engine.py` + `correlation_table.py` | New `nmr_correlation_table.py` mapping chemical shift ranges + multiplicity to proton/carbon environments |
| Output | Functional group + evidence peaks | Proton/carbon environment + evidence peaks, eventually combined with IR output for joint structure hints |

The `Spectrum` dataclass itself is generic (just x/y arrays + metadata) and
should be reusable as-is, with `wavenumbers` conceptually renamed/aliased to
a generic `x_axis` when the codebase is refactored for multi-modality.

## What's genuinely different about NMR (not just a find-and-replace)

- **Integration matters** — relative peak area tells you proton count, which
  IR has no equivalent of. This needs a new feature extractor.
- **Splitting patterns (multiplicity)** carry structural information (n+1
  rule) that has no IR analog — this is arguably NMR's most powerful signal
  and deserves its own feature/model layer, not a shoehorned reuse of the
  IR peak-shape classifier.
- **Reference/solvent peaks** (e.g. TMS, residual CDCl₃) need to be
  identified and excluded — an NMR-specific preprocessing step.
- **¹H vs. ¹³C** are different enough (shift ranges, typical peak count,
  splitting behavior) that they likely warrant separate correlation tables
  even though they share the same pipeline skeleton.

## Suggested build order

1. `¹H NMR` chemical-shift correlation table (chemistry track) — the NMR
   equivalent of `docs/correlation-table.md`.
2. NMR ingestion parser (start with JCAMP-DX, since the format already
   supports NMR data and the ingestion layer's parser structure is reusable).
3. Peak/multiplicity feature extraction.
4. NMR rule engine, mirroring `rule_engine.py`'s evidence-tracing design.
5. ¹³C NMR as a second correlation table once ¹H is validated.
6. Multi-modal fusion layer: given both an IR and NMR rule-engine output for
   the same compound, merge/cross-validate functional-group hypotheses.

No code exists yet for this phase — this file is the starting spec.
