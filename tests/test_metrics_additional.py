from rofa.core.metrics import _correct_fraction, r_w_other_class, top2_coverage


def test_top2_coverage_ignores_invalid_and_none():
    preds = ["A", None, "B", "C", "B", "E", "Z"]
    assert top2_coverage(preds, "B") is True
    assert top2_coverage(preds, "D") is False


def test_top2_coverage_no_valid_predictions():
    preds = [None, "", "E", "1"]
    assert top2_coverage(preds, "A") is False


def test_correct_fraction_counts_only_valid():
    preds = ["A", None, "B", "A", "E", "A"]
    assert _correct_fraction(preds, "A") == 3 / 4
    assert _correct_fraction([], "A") == 0.0


def test_r_w_other_class_bins():
    assert r_w_other_class(0.8, True) == "R"
    assert r_w_other_class(0.85, False) == "W"
    assert r_w_other_class(0.79, True) == "Other"
    assert r_w_other_class(0.9, None) == "Other"
