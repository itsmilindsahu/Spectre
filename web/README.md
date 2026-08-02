# Spectre Web Tool

A self-contained, client-side front end for the Spectre Phase 1 rule engine —
lets any professor in the IISER network run a spectrum through Spectre from
a browser, no install required.

## Deploy (GitHub Pages)

1. Clone the repository from `https://github.com/itsmilindsahu/Spectre.git` and publish it to GitHub Pages.
2. The repository root now includes an `index.html` that redirects to the app under `web/`, so the site opens at `https://itsmilindsahu.github.io/Spectre/` without the `/web/` folder in the URL.
3. No build step or dependencies are required.

## What it does

- Prof enters name + institute (7 IISERs + "Other").
- Uploads a two-column (wavenumber, absorbance) CSV/TSV spectrum, or tries
  one of two embedded example spectra (same synthetic examples as
  `examples/`, downsampled and inlined for portability).
- Runs a JS port of the pipeline in `spectre/preprocessing/`,
  `spectre/features/peaks.py`, and `spectre/models/` entirely in the
  browser — plot, peak table, predicted functional groups with confidence
  and evidence, mirroring `cli.py --json` output.

## Logging ("excel sheet")

- Every run is always logged to the visiting browser's `localStorage`.
  Anyone can hit **Download local log (.csv)** to export what happened in
  their own browser.
- To collect submissions from everyone into one shared Google Sheet, see
  `sheet-logger.gs` and the "Admin setup" panel at the bottom of the page —
  it has the full deployment walkthrough. Once deployed, paste the web app
  URL into the `SHEET_ENDPOINT` constant near the top of the `<script>`
  block in `index.html`.

## Known limitations vs. the Python pipeline

- Baseline correction is a simplified rolling-minimum approximation, not
  the backend's full ALS (asymmetric least squares) solve in
  `preprocessing/baseline.py` — fine for demo/teaching spectra, not a
  byte-for-byte port.
- Phase 1 (rule engine) only, per the roadmap in the top-level `README.md`
  — no ML/CNN/NMR modules.
