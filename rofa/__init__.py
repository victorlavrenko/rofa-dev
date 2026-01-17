"""ROFA package."""

from rofa.core.parse import (
    ParseDebug,
    cop_to_letter,
    extract_choice_letter,
    extract_choice_letter_debug,
)


def __getattr__(name: str):
    if name in {"SYSTEM", "build_user"}:
        from rofa.papers.from_answers_to_hypotheses.prompts import SYSTEM, build_user

        return SYSTEM if name == "SYSTEM" else build_user
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "extract_choice_letter",
    "extract_choice_letter_debug",
    "ParseDebug",
    "cop_to_letter",
    "SYSTEM",
    "build_user",
]
