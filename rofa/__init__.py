"""ROFA package."""

from .prompts import SYSTEM, build_user
from .parse import extract_choice_letter, extract_choice_letter_debug, cop_to_letter

__all__ = [
    "SYSTEM",
    "build_user",
    "extract_choice_letter",
    "extract_choice_letter_debug",
    "cop_to_letter",
]
