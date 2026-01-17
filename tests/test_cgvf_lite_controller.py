from __future__ import annotations

from typing import Dict, Tuple

from rofa.core.cgvf_lite import CgvfLiteConfig, VerifyResult, run_cgvf_lite


def _branches(preds):
    branches = []
    for idx, pred in enumerate(preds):
        branches.append(
            {
                "branch": idx,
                "pred": pred,
                "model_output": f"Explanation: Branch {idx} supports {pred}. Answer: {pred}",
            }
        )
    return branches


class FakeVerifier:
    def __init__(self, mapping: Dict[Tuple[str, int], VerifyResult | None]):
        self.mapping = mapping
        self.calls: list[Tuple[str, int]] = []

    def __call__(self, candidate: str, seed: int, rewritten_summary: str):
        self.calls.append((candidate, seed))
        return self.mapping.get((candidate, seed))


OPTIONS = {"A": "Option A text", "B": "Option B text", "C": "Option C text", "D": "Option D text"}
QUESTION = "Which option is correct?"


def test_high_consensus_returns_top1_without_calls():
    preds = ["A"] * 9 + ["B"]
    verifier = FakeVerifier({})
    final_pred, log = run_cgvf_lite(
        question=QUESTION,
        options=OPTIONS,
        branches=_branches(preds),
        verify_fn=verifier,
        config=CgvfLiteConfig(),
    )

    assert final_pred == "A"
    assert log["final_pred"] == "A"
    assert verifier.calls == []


def test_low_consensus_flip_with_stable_vf_support():
    preds = ["A"] * 6 + ["B"] * 4
    verifier = FakeVerifier(
        {
            ("A", 1): VerifyResult(is_correct=False, final="B", reason="Wrong"),
            ("B", 1): VerifyResult(is_correct=True, final="B", reason="Right"),
            ("A", 2): VerifyResult(is_correct=False, final="B", reason="Wrong"),
            ("B", 2): VerifyResult(is_correct=True, final="B", reason="Right"),
        }
    )
    final_pred, log = run_cgvf_lite(
        question=QUESTION,
        options=OPTIONS,
        branches=_branches(preds),
        verify_fn=verifier,
        config=CgvfLiteConfig(),
    )

    assert final_pred == "B"
    assert log["flipped"] is True


def test_low_consensus_vf_unstable_returns_top1():
    preds = ["A"] * 6 + ["B"] * 4
    verifier = FakeVerifier(
        {
            ("A", 1): VerifyResult(is_correct=False, final="B", reason="Wrong"),
            ("B", 1): VerifyResult(is_correct=True, final="B", reason="Right"),
            ("A", 2): VerifyResult(is_correct=False, final="A", reason="Changed"),
            ("B", 2): VerifyResult(is_correct=True, final="B", reason="Right"),
        }
    )
    final_pred, log = run_cgvf_lite(
        question=QUESTION,
        options=OPTIONS,
        branches=_branches(preds),
        verify_fn=verifier,
        config=CgvfLiteConfig(),
    )

    assert final_pred == "A"
    assert log["flipped"] is False


def test_low_consensus_vf_points_to_top3_returns_top1():
    preds = ["A"] * 6 + ["B"] * 4
    verifier = FakeVerifier(
        {
            ("A", 1): VerifyResult(is_correct=False, final="C", reason="Other"),
            ("B", 1): VerifyResult(is_correct=False, final="C", reason="Other"),
            ("A", 2): VerifyResult(is_correct=False, final="C", reason="Other"),
            ("B", 2): VerifyResult(is_correct=False, final="C", reason="Other"),
        }
    )
    final_pred, log = run_cgvf_lite(
        question=QUESTION,
        options=OPTIONS,
        branches=_branches(preds),
        verify_fn=verifier,
        config=CgvfLiteConfig(),
    )

    assert final_pred == "A"
    assert log["flipped"] is False


def test_low_consensus_vf_parse_failure_returns_top1():
    preds = ["A"] * 6 + ["B"] * 4
    verifier = FakeVerifier({("A", 1): None, ("B", 1): None, ("A", 2): None, ("B", 2): None})
    final_pred, log = run_cgvf_lite(
        question=QUESTION,
        options=OPTIONS,
        branches=_branches(preds),
        verify_fn=verifier,
        config=CgvfLiteConfig(),
    )

    assert final_pred == "A"
    assert log["flipped"] is False
