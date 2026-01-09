"""Answer extraction utilities (unchanged from the notebook)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

# Precompiled regexes
_STRONG_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:final\s+answer|answer)\s*[:\-–]\s*([ABCD])\b", re.I),
    re.compile(r"\b(?:so,?\s*)?(?:the\s+)?answer\s+is\s*([ABCD])\b", re.I),
    re.compile(r"\b(?:correct\s+answer|correct\s+option)\s*(?:is|:)\s*([ABCD])\b", re.I),
    re.compile(
        r"\boption\s*([ABCD])\b\s*(?:is\s*)?(?:correct|best|most\s+likely|most\s+accurate)\b",
        re.I,
    ),
]

_KEYWORD_RX = re.compile(r"\b(answer|final|correct|best|therefore|so)\b", re.I)
_END_LETTER_RX = re.compile(r"\b([ABCD])\b\s*[\.\!\?]?\s*$", re.I)
_PAREN_LETTER_RX = re.compile(r"[\(\[]\s*([ABCD])\s*[\)\]]", re.I)
_ENUM_LINE_RX = re.compile(r"^\s*([ABCD])\s*[\.\)]\s", re.I)
_WEAK_TAIL_RX = re.compile(r"(?:^|[\s\(\[\{])([ABCD])(?:[\]\}\)\s\.\!\?]|$)", re.I)


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


def _extract_strong_match(text: str) -> Optional[Tuple[str, int]]:
    """Return a strong match letter and pattern index if found."""
    last: Optional[Tuple[int, str, int]] = None
    for idx, rx in enumerate(_STRONG_PATTERNS):
        for match in rx.finditer(text):
            last = (match.start(1), match.group(1).upper(), idx)
    if last is None:
        return None
    return last[1], last[2]


def _score_tail(tail: str) -> Dict[str, int]:
    scores: Dict[str, int] = {k: 0 for k in "ABCD"}

    for match in _KEYWORD_RX.finditer(tail):
        window = tail[match.start() : match.start() + 120]
        for letter in "ABCD":
            if re.search(rf"\b{letter}\b", window, re.I):
                scores[letter] += 5

    end_match = _END_LETTER_RX.search(tail)
    if end_match:
        scores[end_match.group(1).upper()] += 6

    for match in _PAREN_LETTER_RX.finditer(tail):
        scores[match.group(1).upper()] += 2

    enum_counts = {k: 0 for k in "ABCD"}
    for line in tail.splitlines():
        enum_match = _ENUM_LINE_RX.match(line)
        if enum_match:
            enum_counts[enum_match.group(1).upper()] += 1

    if sum(enum_counts.values()) >= 2:
        for letter in "ABCD":
            scores[letter] -= min(2, enum_counts[letter])

    return scores


def _select_best_score(scores: Dict[str, int]) -> Tuple[str, int]:
    return max(scores.items(), key=lambda kv: kv[1])


def _extract_weak_tail(tail: str) -> Optional[str]:
    last_letter: Optional[str] = None
    for match in _WEAK_TAIL_RX.finditer(tail):
        letter = match.group(1).upper()
        suffix = tail[match.start() : match.start() + 4]
        if _ENUM_LINE_RX.match(suffix):
            continue
        last_letter = letter
    return last_letter


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

    strong = _extract_strong_match(t)
    if strong is not None:
        letter, pattern_idx = strong
        dbg = (
            ExtractDebug(method=f"strong[{pattern_idx}]", scores={letter: 999})
            if return_debug
            else None
        )
        return letter, dbg

    tail = t[-800:]
    scores = _score_tail(tail)
    best_letter, best_score = _select_best_score(scores)
    if best_score >= 4:
        dbg = ExtractDebug(method="tail-score", scores=scores) if return_debug else None
        return best_letter, dbg

    last_letter = _extract_weak_tail(tail)
    dbg = ExtractDebug(method="weak-tail", scores=scores) if return_debug else None
    return last_letter, dbg


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
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


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
