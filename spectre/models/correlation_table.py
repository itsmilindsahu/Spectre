"""
correlation_table.py

The domain-knowledge core of Spectre's rule engine: a structured version of
the classic IR peak-region correlation table. This is the chemistry track's
primary artifact -- every row here should be reviewed/extended by a chemist
(Paarth), not just the computational track.

Each rule maps a wavenumber region + expected peak shape to a candidate
functional group. The rule engine (rule_engine.py) matches detected peaks
against these rows and reports which rule(s) fired as evidence.

Fields:
    low, high      : wavenumber range (cm^-1) this rule applies to
    shape           : expected peak shape - "broad", "sharp", "medium", "variable"
    min_width       : minimum peak width (cm^-1) to count as "broad" (rough heuristic)
    bond            : the vibrating bond responsible for the signal
    group           : candidate functional group(s)
    note            : human-readable interpretation note
    priority        : used to break ties when multiple rules match the same peak
                       (higher = more diagnostic / more likely correct)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CorrelationRule:
    low: float
    high: float
    shape: str  # "broad" | "sharp" | "medium" | "variable"
    bond: str
    group: str
    note: str
    priority: int = 1
    min_width: Optional[float] = None  # cm^-1, only meaningful for "broad" rules


CORRELATION_TABLE = [
    # --- Region 1: O-H / N-H stretch zone (3600-3200) ---
    CorrelationRule(
        low=3600, high=3200, shape="broad", bond="O-H stretch",
        group="Alcohol / Phenol",
        note="Broad, smooth, u-shaped curve ('tongue'). Free/H-bonded O-H stretch.",
        priority=3, min_width=80,
    ),
    CorrelationRule(
        low=3400, high=3250, shape="sharp", bond="N-H stretch",
        group="Amine / Amide",
        note="Sharper and weaker than O-H. Primary amine: two peaks. "
             "Secondary amine/amide: one peak.",
        priority=3,
    ),
    CorrelationRule(
        low=3300, high=2500, shape="broad", bond="O-H stretch (H-bonded dimer)",
        group="Carboxylic Acid",
        note="Extremely broad, 'messy valley' centered near 3000, often overlaps "
             "the C-H stretch region entirely.",
        priority=4, min_width=400,
    ),

    # --- Region 2: C-H stretch zone, split by the 3000 cm-1 boundary ---
    CorrelationRule(
        low=3333, high=3267, shape="sharp", bond="sp C-H stretch",
        group="Terminal Alkyne",
        note="Sharp peak right around 3300 cm^-1.",
        priority=3,
    ),
    CorrelationRule(
        low=3100, high=3010, shape="sharp", bond="sp2 C-H stretch",
        group="Alkene / Aromatic ring",
        note="Sharp peak just above the 3000 cm^-1 boundary line.",
        priority=2,
    ),
    CorrelationRule(
        low=2950, high=2850, shape="sharp", bond="sp3 C-H stretch",
        group="Alkane",
        note="Sharp, jagged peak(s) just below the 3000 cm^-1 boundary line. "
             "Present in almost all organic compounds -- low diagnostic value alone.",
        priority=1,
    ),

    # --- Region 3: triple bonds ---
    CorrelationRule(
        low=2260, high=2100, shape="sharp", bond="C#C or C#N stretch",
        group="Alkyne / Nitrile",
        note="Thin, short, weak spike. Easy to miss -- scan carefully.",
        priority=2,
    ),

    # --- Region 4: carbonyl / alkene "sword" zone ---
    CorrelationRule(
        low=1760, high=1665, shape="sharp", bond="C=O stretch",
        group="Carbonyl (Ketone / Aldehyde / Ester / Carboxylic Acid)",
        note="Strong, sharp, deep peak -- usually the most diagnostic single peak "
             "in the whole spectrum.",
        priority=5,
    ),
    CorrelationRule(
        low=1680, high=1640, shape="medium", bond="C=C stretch",
        group="Alkene",
        note="Much weaker and narrower than the carbonyl peak nearby.",
        priority=2,
    ),
]


def rule_contains(rule: CorrelationRule, wavenumber: float) -> bool:
    """
    Check whether a wavenumber falls inside a rule's range.

    NOTE: by IR convention, rule.low holds the numerically LARGER wavenumber
    (high-energy end) and rule.high holds the numerically SMALLER one, e.g.
    low=3600, high=3200 means the range 3200-3600 cm^-1.
    """
    return rule.high <= wavenumber <= rule.low
