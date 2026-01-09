"""Metric computations for ROFA experiments."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Optional, Sequence, Tuple, TypedDict


class DiversityMetrics(TypedDict):
    """Structured diversity metrics for branch predictions."""

    leader: Optional[str]
    max_frac: float
    variation_ratio: float
    entropy_bits: float
    valid_n: int
    none_n: int
    unanimous: bool
    unanim_valid: bool


def _safe_counter_preds(preds: Iterable[Optional[str]]) -> Tuple[Counter, int]:
    """Return counts of valid letters and the valid count."""
    letters = [p for p in preds if p in ("A", "B", "C", "D")]
    return Counter(letters), len(letters)


def _entropy_from_counter(cnt: Counter, total: int) -> float:
    """Compute Shannon entropy in bits from a counter."""
    if total <= 0:
        return 0.0
    ent = 0.0
    for _, v in cnt.items():
        p = v / total
        ent -= p * math.log(p + 1e-12, 2)
    return ent


def _diversity_metrics(preds: Sequence[Optional[str]]) -> DiversityMetrics:
    """Compute diversity metrics for branch predictions.

    Args:
        preds: List of branch predictions (A/B/C/D or None).

    Returns:
        A dictionary with leader, max_frac, entropy, and unanimity statistics:
        ``leader``, ``max_frac``, ``variation_ratio``, ``entropy_bits``, ``valid_n``,
        ``none_n``, ``unanimous``, and ``unanim_valid``.
    """
    cnt, valid_n = _safe_counter_preds(preds)
    none_n = len(preds) - valid_n

    if valid_n == 0:
        return {
            "leader": None,
            "max_frac": 0.0,
            "variation_ratio": 0.0,
            "entropy_bits": 0.0,
            "valid_n": 0,
            "none_n": none_n,
            "unanimous": False,
            "unanim_valid": False,
        }

    leader, leader_count = cnt.most_common(1)[0]
    max_frac = leader_count / valid_n
    variation_ratio = 1.0 - max_frac
    entropy_bits = _entropy_from_counter(cnt, valid_n)

    unanim_valid = len(cnt) == 1
    unanimous = unanim_valid and valid_n == len(preds)

    return {
        "leader": leader,
        "max_frac": max_frac,
        "variation_ratio": variation_ratio,
        "entropy_bits": entropy_bits,
        "valid_n": valid_n,
        "none_n": none_n,
        "unanimous": unanimous,
        "unanim_valid": unanim_valid,
    }


def _correct_fraction(preds: Sequence[Optional[str]], gold: str) -> float:
    """Return fraction of valid predictions equal to gold."""
    valid = [p for p in preds if p in ("A", "B", "C", "D")]
    if not valid:
        return 0.0
    return sum(1 for p in valid if p == gold) / len(valid)


def top2_coverage(preds: Sequence[Optional[str]], gold: str) -> bool:
    """Return True if gold is among the top-2 most frequent predictions.

    Args:
        preds: Branch predictions (A/B/C/D or None).
        gold: Gold label.

    Returns:
        True if gold is in the top-2 most frequent predictions.

    Stability:
        This definition matches the baseline paper and should remain unchanged across
        papers unless a protocol version is incremented.
    """
    cnt, valid_n = _safe_counter_preds(preds)
    if valid_n == 0:
        return False
    top_two = [item[0] for item in cnt.most_common(2)]
    return gold in top_two


def r_w_other_class(max_frac: float, leader_correct: Optional[bool]) -> str:
    """Classify based on consensus bins used in the original notebook.

    Stability:
        This binning matches the baseline paper and should not change without a
        protocol version bump.
    """
    if leader_correct is None:
        return "Other"
    if max_frac >= 0.8:
        return "R" if leader_correct else "W"
    return "Other"
