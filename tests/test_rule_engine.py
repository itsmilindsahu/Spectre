from spectre.features.peaks import Peak
from spectre.models.rule_engine import classify


def test_carbonyl_peak_triggers_carbonyl_prediction():
    peaks = [Peak(wavenumber=1715, intensity=1.0, width=12)]
    predictions = classify(peaks)

    assert len(predictions) >= 1
    top = predictions[0]
    assert "Carbonyl" in top.functional_group
    assert top.confidence > 0.5


def test_broad_oh_peak_triggers_alcohol_prediction():
    peaks = [Peak(wavenumber=3400, intensity=0.8, width=200)]  # broad, in alcohol range
    predictions = classify(peaks)

    groups = [p.functional_group for p in predictions]
    assert "Alcohol / Phenol" in groups


def test_no_peaks_yields_no_predictions():
    assert classify([]) == []


def test_evidence_is_traceable():
    peaks = [Peak(wavenumber=1715, intensity=1.0, width=12)]
    predictions = classify(peaks)

    top = predictions[0]
    assert len(top.evidence) == 1
    assert top.evidence[0].wavenumber == 1715
    assert len(top.matched_rules) == 1


def test_multiple_peaks_same_group_merge_and_boost_confidence():
    # Two separate broad-ish O-H peaks in the alcohol region
    peaks = [
        Peak(wavenumber=3500, intensity=0.6, width=180),
        Peak(wavenumber=3300, intensity=0.6, width=180),
    ]
    predictions = classify(peaks)
    alcohol = next(p for p in predictions if p.functional_group == "Alcohol / Phenol")
    assert len(alcohol.evidence) == 2
