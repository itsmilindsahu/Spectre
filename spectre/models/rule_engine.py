"""
rule_engine.py

Phase 1 model: a transparent, rule-based functional-group classifier.
Every prediction is traceable back to the exact peak(s) and correlation-table
rule that produced it -- this is the baseline that classical ML (Phase 2)
and deep learning (Phase 3) models must beat, and it stays in the pipeline
permanently as an interpretable cross-check even after ML models are added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from spectre.features.peaks import Peak
from spectre.models.correlation_table import CORRELATION_TABLE, CorrelationRule, rule_contains


@dataclass
class Prediction:
    functional_group: str
    confidence: float
    evidence: List[Peak] = field(default_factory=list)
    matched_rules: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "functional_group": self.functional_group,
            "confidence": round(self.confidence, 3),
            "evidence_peaks_cm-1": [round(p.wavenumber, 1) for p in self.evidence],
            "notes": self.matched_rules,
        }


def _shape_match_score(peak: Peak, rule: CorrelationRule) -> float:
    """
    Score how well a peak's observed shape matches a rule's expected shape.
    Exact match scores highest; "variable"-shape rules always partially match
    since they don't constrain shape.
    """
    if rule.shape == "variable":
        return 0.7
    if peak.shape == rule.shape:
        return 1.0
    # Adjacent categories (e.g. medium vs sharp) still get partial credit
    order = ["sharp", "medium", "broad"]
    try:
        dist = abs(order.index(peak.shape) - order.index(rule.shape))
    except ValueError:
        return 0.5
    return max(0.3, 1.0 - 0.35 * dist)


def classify(peaks: List[Peak]) -> List[Prediction]:
    """
    Run the rule engine over a list of detected peaks and return
    functional-group predictions, sorted by confidence descending.

    Multiple peaks supporting the same functional group are merged into a
    single Prediction with combined evidence.
    """
    group_hits: dict[str, Prediction] = {}

    for peak in peaks:
        for rule in CORRELATION_TABLE:
            if not rule_contains(rule, peak.wavenumber):
                continue

            shape_score = _shape_match_score(peak, rule)
            # Confidence blends: how diagnostic the rule is (priority),
            # how well the peak shape matches, and the peak's own intensity
            confidence = min(1.0, (rule.priority / 5.0) * shape_score * (0.5 + 0.5 * peak.intensity))

            pred = group_hits.get(rule.group)
            if pred is None:
                pred = Prediction(functional_group=rule.group, confidence=0.0)
                group_hits[rule.group] = pred

            pred.evidence.append(peak)
            pred.matched_rules.append(
                f"{rule.bond} near {peak.wavenumber:.0f} cm^-1 ({rule.note})"
            )
            # Take the strongest single piece of evidence as the group's confidence,
            # but nudge upward slightly for corroborating peaks (diminishing returns)
            pred.confidence = max(pred.confidence, confidence) + (
                0.05 * (len(pred.evidence) - 1) if len(pred.evidence) > 1 else 0.0
            )
            pred.confidence = min(pred.confidence, 0.99)

    predictions = sorted(group_hits.values(), key=lambda p: p.confidence, reverse=True)
    return predictions
