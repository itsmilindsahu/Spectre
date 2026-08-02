# IR Correlation Table Reference

This is the human-readable version of `spectre/models/correlation_table.py` —
the source of truth is the code file (the rule engine imports it directly),
but this document exists so the chemistry track can review/propose changes
without reading Python.

**If you (Paarth) want to change a rule: edit this table's description here
first for review, then port the change into `correlation_table.py`.**

| Wavenumber Region (cm⁻¹) | Shape | Bond | Functional Group | Priority |
|---|---|---|---|---|
| 3600–3200 | Broad | O-H stretch | Alcohol / Phenol | 3 |
| 3400–3250 | Sharp | N-H stretch | Amine / Amide | 3 |
| 3300–2500 | Broad (very) | O-H stretch (H-bonded) | Carboxylic Acid | 4 |
| 3333–3267 | Sharp | sp C-H stretch | Terminal Alkyne | 3 |
| 3100–3010 | Sharp | sp² C-H stretch | Alkene / Aromatic ring | 2 |
| 2950–2850 | Sharp | sp³ C-H stretch | Alkane | 1 |
| 2260–2100 | Sharp (weak) | C≡C or C≡N stretch | Alkyne / Nitrile | 2 |
| 1760–1665 | Sharp (strong) | C=O stretch | Carbonyl (ketone/aldehyde/ester/acid) | 5 |
| 1680–1640 | Medium | C=C stretch | Alkene | 2 |

## Priority scale

Priority determines how much weight a matched rule carries when multiple
functional groups could explain the same peak (used in `rule_engine.py`'s
confidence scoring). Higher priority = historically more diagnostic /
less ambiguous.

- **5** — essentially unambiguous when present (carbonyl)
- **4** — very distinctive shape, hard to confuse (carboxylic acid's broad valley)
- **3** — distinctive but occasionally confusable with a neighboring rule
- **2** — useful corroborating evidence, moderate ambiguity
- **1** — present in nearly everything, low standalone diagnostic value (alkane C-H)

## Known limitations (please extend this list)

- The fingerprint region (<1500 cm⁻¹) is intentionally not covered by
  peak-region rules — it's too dense and compound-specific for a simple
  correlation table. It's reserved for a future spectral-matching module
  (comparing the whole fingerprint region against a reference library)
  rather than rule-based peak interpretation.
- Overlapping regions (e.g. carboxylic acid's broad O-H spans directly over
  the C-H stretch region) mean a single peak can trigger multiple candidate
  labels — this is intentional and mirrors real ambiguity a chemist faces,
  but it means the rule engine's output should be read as "candidates with
  evidence," not a single definitive answer.
- Ester vs. ketone vs. aldehyde vs. acid are currently lumped into one
  "Carbonyl" label since they share the same core C=O stretch region.
  Distinguishing them requires secondary peaks (e.g. C-O stretch for esters,
  aldehyde C-H doublet near 2820/2720) that aren't yet in the table —
  a good first contribution for the chemistry track.
- Conjugation, ring strain, and electronic effects can shift any of these
  ranges by 10-30 cm⁻¹ in real compounds. The ranges here are textbook
  central tendencies, not hard physical boundaries.

## Adding a new rule

1. Add a `CorrelationRule(...)` entry to `CORRELATION_TABLE` in
   `spectre/models/correlation_table.py`.
2. Add the same row to the table above.
3. Add at least one test case in `tests/test_rule_engine.py` that exercises
   the new rule.
