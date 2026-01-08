import math

from rofa.metrics import _diversity_metrics, _entropy_from_counter, _safe_counter_preds


def test_diversity_metrics_unanimous():
    preds = ["A"] * 10
    metrics = _diversity_metrics(preds)
    assert metrics["leader"] == "A"
    assert metrics["max_frac"] == 1.0
    assert metrics["variation_ratio"] == 0.0
    assert metrics["entropy_bits"] == 0.0
    assert metrics["valid_n"] == 10
    assert metrics["none_n"] == 0
    assert metrics["unanimous"] is True


def test_diversity_metrics_mixed_with_none():
    preds = ["A", "B", "A", None, "C", "A", "B", None, "A", "D"]
    metrics = _diversity_metrics(preds)
    assert metrics["leader"] == "A"
    assert metrics["valid_n"] == 8
    assert metrics["none_n"] == 2
    assert math.isclose(metrics["max_frac"], 4 / 8)
    assert math.isclose(metrics["variation_ratio"], 1 - 4 / 8)


def test_entropy_matches_counter():
    cnt, total = _safe_counter_preds(["A", "A", "B", "B"])
    ent = _entropy_from_counter(cnt, total)
    assert math.isclose(ent, 1.0, rel_tol=1e-6)
