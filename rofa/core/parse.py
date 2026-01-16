"""Answer extraction utilities (unchanged from the notebook)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

# Precompiled regexes
_RULE1_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"^\s*final\s*[:\-]\s*([ABCD])\b", re.I | re.M), 1),
    (re.compile(r"\bfinal\s+answer\s*[:\-]\s*([ABCD])\b", re.I), 1),
    (re.compile(r"\banswer\s*[:\-]\s*([ABCD])\b", re.I), 1),
    (re.compile(r"\banswer\s+is\s*[:\-]?\s*[*_]*([ABCD])\b", re.I), 1),
    (
        re.compile(r"\b(?:correct\s+answer\s+is|the\s+correct\s+answer\s+is)\s*([ABCD])\b", re.I),
        1,
    ),
    (
        re.compile(
            r"\boption\s*([ABCD])\b\s*(?:is\s*)?(?:correct|best|most\s+likely|most\s+accurate)\b",
            re.I,
        ),
        1,
    ),
    (re.compile(r"\(([ABCD])\)\s+is\s+(?:best|correct)\b", re.I), 1),
    (re.compile(r"\bwe\s+choose\s+([ABCD])\b", re.I), 1),
]

_LEADING_LINE_RX = re.compile(r"^\s*([ABCD])\s*[\.\)\:\-]\s+", re.I)
_TRAILING_THEREFORE_RX = re.compile(
    r"\b(therefore|so|thus)\b.*\b([ABCD])\b\s*[\.\)]?\s*$",
    re.I | re.S,
)
_TRAILING_ANSWER_RX = re.compile(r"\banswer\s+is\s*[:\-]?\s*[*_]*([ABCD])\b\s*$", re.I)
_STANDALONE_LINE_RX = re.compile(r"^\s*([ABCD])\s*[\.]?\s*$", re.I)
_CORRECT_ANSWER_OVERRIDE_RX = re.compile(
    r"\b(?:therefore,\s*)?(?:the\s*)?correct answer is\s*([ABCD])\b",
    re.I,
)
_FUZZY_TOKEN_RX = re.compile(r"[a-z0-9/\.]+", re.I)
_FUZZY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "per",
    "that",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class ExtractDebug:
    """Debug information for choice extraction."""

    method: str
    scores: Dict[str, int]


@dataclass(frozen=True)
class ParseDebug:
    """Structured debug payload for answer extraction."""

    letter: Optional[str]
    scores: Dict[str, int]
    method: str
    matched_option: Optional[str] = None


def _extract_rule1(text: str) -> Optional[Tuple[str, int]]:
    """Return letter and pattern index for explicit final markers."""
    for idx, (rx, group_index) in enumerate(_RULE1_PATTERNS):
        last_match: Optional[re.Match[str]] = None
        for match in rx.finditer(text):
            last_match = match
        if last_match is not None:
            return last_match.group(group_index).upper(), idx
    return None


def _extract_rule2(text: str) -> Optional[str]:
    """Return leading choice label from the first non-empty line."""
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _LEADING_LINE_RX.match(line)
        if match:
            return match.group(1).upper()
        return None
    return None


def _extract_rule3(text: str) -> Optional[Tuple[str, str]]:
    """Return trailing conclusion patterns and the method label."""
    tail = text[-400:]
    match = _TRAILING_THEREFORE_RX.search(tail)
    if match:
        return match.group(2).upper(), "therefore"
    match = _TRAILING_ANSWER_RX.search(tail)
    if match:
        return match.group(1).upper(), "answer"
    return None


def _extract_rule4(text: str, *, lines: int = 5) -> Optional[str]:
    """Return a conservative fallback from the last non-empty lines."""
    non_empty = [line for line in text.splitlines() if line.strip()]
    for line in reversed(non_empty[-lines:]):
        match = _STANDALONE_LINE_RX.match(line)
        if match:
            return match.group(1).upper()
    return None


def _extract_impl(
    text: str,
    *,
    return_debug: bool = False,
) -> Tuple[Optional[str], Optional[ExtractDebug]]:
    """Return the extracted letter and optional debug metadata."""
    if not text:
        dbg = ExtractDebug(method="empty", scores={}) if return_debug else None
        return None, dbg

    t = text.strip()

    strong = _extract_rule1(t)
    if strong is not None:
        letter, pattern_idx = strong
        dbg = (
            ExtractDebug(method=f"rule1[{pattern_idx}]", scores={letter: 1})
            if return_debug
            else None
        )
        return letter, dbg

    leading = _extract_rule2(t)
    if leading is not None:
        dbg = ExtractDebug(method="rule2", scores={leading: 1}) if return_debug else None
        return leading, dbg

    trailing = _extract_rule3(t)
    if trailing is not None:
        letter, method = trailing
        dbg = ExtractDebug(method=f"rule3[{method}]", scores={letter: 1}) if return_debug else None
        return letter, dbg

    fallback = _extract_rule4(t)
    dbg = ExtractDebug(method="rule4", scores={}) if return_debug else None
    return fallback, dbg


def _coerce_options(options: Mapping[str, str] | Sequence[str]) -> Dict[str, str]:
    if isinstance(options, Mapping):
        return {str(k).upper(): str(v) for k, v in options.items() if k is not None}
    if isinstance(options, Sequence):
        option_list = list(options)
        if len(option_list) == 4:
            return {
                letter: str(value)
                for letter, value in zip("ABCD", option_list, strict=False)
            }
    return {}


def _normalize_option_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _option_is_too_short(option: str) -> bool:
    if len(option) >= 3:
        return False
    return not re.search(r"\d", option)


def _fuzzy_tokens(text: str) -> set[str]:
    tokens = _FUZZY_TOKEN_RX.findall(text.lower())
    filtered = set()
    for token in tokens:
        token = token.strip(".")
        if not token:
            continue
        if token in _FUZZY_STOPWORDS:
            continue
        if len(token) >= 3 or re.search(r"[\d/]", token):
            filtered.add(token)
    return filtered


def _has_numeric_token(tokens: set[str]) -> bool:
    return any(re.search(r"[\d/]", token) for token in tokens)


def _match_option_text(text: str, options: Mapping[str, str]) -> Optional[str]:
    if not text:
        return None
    tail_norm = _normalize_option_text(text[-800:])
    if not tail_norm:
        return None

    candidates = []
    for letter, option_text in options.items():
        option_norm = _normalize_option_text(option_text)
        if not option_norm or _option_is_too_short(option_norm):
            continue
        idx = tail_norm.rfind(option_norm)
        if idx != -1:
            candidates.append((len(option_norm), idx, letter))

    if not candidates:
        output_tokens = _fuzzy_tokens(text[-800:])
        if not output_tokens:
            return None
        fuzzy_candidates = []
        for letter, option_text in options.items():
            option_norm = _normalize_option_text(option_text)
            if not option_norm or _option_is_too_short(option_norm):
                continue
            option_tokens = _fuzzy_tokens(option_text)
            if not option_tokens:
                continue
            overlap = output_tokens & option_tokens
            overlap_count = len(overlap)
            numeric_overlap = _has_numeric_token(overlap)
            if overlap_count >= 2 or (overlap_count == 1 and numeric_overlap):
                fuzzy_candidates.append(
                    (overlap_count, int(numeric_overlap), len(option_tokens), letter)
                )

        if not fuzzy_candidates:
            return None
        fuzzy_candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        if (
            len(fuzzy_candidates) > 1
            and fuzzy_candidates[0][:3] == fuzzy_candidates[1][:3]
        ):
            return None
        return fuzzy_candidates[0][3]

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _extract_correct_answer_override(text: str) -> Optional[str]:
    """Return the explicit correct-answer override if present."""
    last_match: Optional[re.Match[str]] = None
    for match in _CORRECT_ANSWER_OVERRIDE_RX.finditer(text):
        last_match = match
    if last_match is None:
        return None
    return last_match.group(1).upper()


def extract_choice_letter(
    text: str,
    options: Mapping[str, str] | Sequence[str] | None = None,
) -> Optional[str]:
    """Extract a single answer choice letter from model output.

    Args:
        text: Model output text.
        options: Optional mapping or list of options for fallback matching.

    Returns:
        The extracted letter (A/B/C/D) or None.

    Failure modes:
        Returns None if no letter can be inferred, or if output is empty/ambiguous.

    Stability:
        This parser is part of the stable core protocol; papers should avoid modifying
        its heuristics without documenting a protocol change.
    """
    letter, _ = _extract_impl(text, return_debug=False)
    override = _extract_correct_answer_override(text)
    if override is not None:
        return override
    if letter is not None or not options:
        return letter
    option_map = _coerce_options(options)
    if not option_map:
        return letter
    return _match_option_text(text, option_map) or letter


def extract_choice_letter_debug(
    text: str,
    options: Mapping[str, str] | Sequence[str] | None = None,
) -> ParseDebug:
    """Return structured debug metadata for answer extraction.

    Args:
        text: Model output text.
        options: Optional mapping or list of options for fallback matching.

    Returns:
        ParseDebug containing the extracted letter, scoring, and method name.

    Failure modes:
        ``method`` may be ``unknown`` if no heuristic fires; ``letter`` may be None if
        no extraction is possible.
    """
    letter, dbg = _extract_impl(text, return_debug=True)
    if letter is None and options:
        option_map = _coerce_options(options)
        matched = _match_option_text(text, option_map) if option_map else None
        if matched is not None:
            scores = dbg.scores if dbg else {}
            return ParseDebug(
                letter=matched,
                scores=scores,
                method="option-text",
                matched_option=matched,
            )
    if dbg is None:
        return ParseDebug(letter=letter, scores={}, method="unknown")
    return ParseDebug(letter=letter, scores=dbg.scores, method=dbg.method)


def cop_to_letter(cop: int):
    """Convert integer choice index to letter (0..3 -> A..D)."""
    return "ABCD"[int(cop)]
