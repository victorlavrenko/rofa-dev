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


def _extract_impl(text: str, *, return_debug: bool = False) -> Tuple[Optional[str], Optional[ExtractDebug]]:
    """Return the extracted letter and optional debug metadata."""
    if not text:
        dbg = ExtractDebug(method="empty", scores={}) if return_debug else None
        return None, dbg

    t = text.strip()

    # 1) Strong patterns: take the last match across all patterns
    last: Optional[Tuple[int, str, int]] = None  # (pos, letter, pattern_idx)
    for idx, rx in enumerate(_STRONG_PATTERNS):
        for m in rx.finditer(t):
            last = (m.start(1), m.group(1).upper(), idx)

    if last is not None:
        letter = last[1]
        dbg = ExtractDebug(method=f"strong[{last[2]}]", scores={letter: 999}) if return_debug else None
        return letter, dbg

    # 2) Tail scoring
    tail = t[-800:]
    scores: Dict[str, int] = {k: 0 for k in "ABCD"}

    for m in _KEYWORD_RX.finditer(tail):
        window = tail[m.start() : m.start() + 120]
        for L in "ABCD":
            if re.search(rf"\b{L}\b", window, re.I):
                scores[L] += 5

    m_end = _END_LETTER_RX.search(tail)
    if m_end:
        scores[m_end.group(1).upper()] += 6

    for m in _PAREN_LETTER_RX.finditer(tail):
        scores[m.group(1).upper()] += 2

    enum_counts = {k: 0 for k in "ABCD"}
    for line in tail.splitlines():
        mm = _ENUM_LINE_RX.match(line)
        if mm:
            enum_counts[mm.group(1).upper()] += 1

    if sum(enum_counts.values()) >= 2:
        for L in "ABCD":
            scores[L] -= min(2, enum_counts[L])

    best_letter, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score >= 4:
        dbg = ExtractDebug(method="tail-score", scores=scores) if return_debug else None
        return best_letter, dbg

    # 3) Last resort: last isolated letter token in tail, avoiding enumeration headers
    last_letter: Optional[str] = None
    for m in _WEAK_TAIL_RX.finditer(tail):
        L = m.group(1).upper()
        suffix = tail[m.start() : m.start() + 4]
        if _ENUM_LINE_RX.match(suffix):
            continue
        last_letter = L

    dbg = ExtractDebug(method="weak-tail", scores=scores) if return_debug else None
    return last_letter, dbg


def _coerce_options(options: Mapping[str, str] | Sequence[str]) -> Dict[str, str]:
    if isinstance(options, Mapping):
        return {str(k).upper(): str(v) for k, v in options.items() if k is not None}
    if isinstance(options, Sequence):
        option_list = list(options)
        if len(option_list) == 4:
            return {letter: str(value) for letter, value in zip("ABCD", option_list)}
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
    """Extract a single answer choice letter from model output."""
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
) -> Tuple[Optional[str], Dict[str, int], str]:
    """Return the extracted letter, score map, and strategy name."""
    letter, dbg = _extract_impl(text, return_debug=True)
    if letter is None and options:
        option_map = _coerce_options(options)
        matched = _match_option_text(text, option_map) if option_map else None
        if matched is not None:
            scores = dbg.scores if dbg else {}
            return matched, scores, "option-text"
    if dbg is None:
        return letter, {}, "unknown"
    return letter, dbg.scores, dbg.method


def cop_to_letter(cop: int):
    """Convert integer choice index to letter (0..3 -> A..D)."""
    return "ABCD"[int(cop)]
