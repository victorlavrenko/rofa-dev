"""Paper configuration for 'From Answers to Hypotheses'."""

from __future__ import annotations

from rofa.core.registry import MethodSpec, PaperSpec, register_paper
from rofa.papers.from_answers_to_hypotheses.methods import BranchSamplingEnsemble, GreedyDecode

PAPER_ID = "from_answers_to_hypotheses"
PAPER_TITLE = (
    "From Answers to Hypotheses: Internal Consensus and Its Limits in Large Language Models"
)
PAPER_DESCRIPTION = (
    "Baseline experiments comparing greedy decoding and parallel branch sampling "
    "for clinical reasoning questions."
)

DEFAULT_DATASET = "openlifescienceai/medmcqa"
DEFAULT_SPLIT = "validation"

METHODS = {
    "greedy": MethodSpec(
        key="greedy",
        factory=GreedyDecode,
        description="Single greedy decoding pass.",
    ),
    "k_sample_ensemble": MethodSpec(
        key="k_sample_ensemble",
        factory=BranchSamplingEnsemble,
        description="Multi-sample ensemble (branch sampling) with consensus metrics.",
    ),
}

PAPER_SPEC = register_paper(
    PaperSpec(
        paper_id=PAPER_ID,
        title=PAPER_TITLE,
        description=PAPER_DESCRIPTION,
        default_dataset=DEFAULT_DATASET,
        default_split=DEFAULT_SPLIT,
        methods=METHODS,
    )
)
