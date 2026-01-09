"""ROFA package."""

from rofa.core.parse import extract_choice_letter, extract_choice_letter_debug, cop_to_letter
from rofa.papers.from_answers_to_hypotheses.prompts import SYSTEM, build_user

__all__ = [
    "SYSTEM",
    "build_user",
    "extract_choice_letter",
    "extract_choice_letter_debug",
    "cop_to_letter",
]
