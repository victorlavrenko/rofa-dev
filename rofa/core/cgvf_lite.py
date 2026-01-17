"""Conservative top-2 flip controller (CGVF-lite)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


VALID_CHOICES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class CgvfLiteConfig:
    """Configuration for CGVF-lite controller."""

    top1_min: int = 3
    top1_max: int = 8
    gap_max: int = 6
    consensus_threshold: float = 0.60
    gap_threshold: int = 2
    max_excerpt_chars: int = 400
    max_excerpts_per_group: int = 2
    min_sentences: int = 2
    max_sentences: int = 4


@dataclass(frozen=True)
class VerifyResult:
    """Parsed verification response."""

    is_correct: bool
    final: str
    reason: str


Verifier = Callable[[str, int, str], Optional[VerifyResult]]


_STOPWORDS = {
    "patient",
    "patients",
    "disease",
    "diseases",
    "condition",
    "conditions",
    "clinical",
    "diagnosis",
    "symptom",
    "symptoms",
    "sign",
    "signs",
    "finding",
    "findings",
    "treatment",
    "therapy",
    "therapies",
    "management",
    "present",
    "presents",
    "presenting",
    "history",
    "likely",
    "common",
    "based",
    "option",
    "options",
    "answer",
    "correct",
    "would",
    "could",
}


def _safe_choice(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str) and value in VALID_CHOICES:
        return value
    return None


def compute_vote_counts(branches: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, int], int]:
    counts = {choice: 0 for choice in VALID_CHOICES}
    valid = 0
    for branch in branches:
        pred = _safe_choice(branch.get("pred"))
        if pred:
            counts[pred] += 1
            valid += 1
    return counts, valid


def rank_labels(counts: Dict[str, int]) -> List[str]:
    return sorted(VALID_CHOICES, key=lambda letter: (-counts.get(letter, 0), letter))


def _low_consensus(consensus: float, gap: int, config: CgvfLiteConfig) -> bool:
    return consensus <= config.consensus_threshold or gap <= config.gap_threshold


def _eligible_vote_pattern(
    counts: Dict[str, int],
    top1: str,
    top2: Optional[str],
    top3: Optional[str],
    config: CgvfLiteConfig,
) -> bool:
    if top2 is None:
        return False
    top1_count = counts.get(top1, 0)
    top2_count = counts.get(top2, 0)
    if top3 is not None and counts.get(top3, 0) >= top2_count:
        return False
    if top1_count < top2_count:
        return False
    if not (config.top1_min <= top1_count <= config.top1_max):
        return False
    if top1_count - top2_count > config.gap_max:
        return False
    return True


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _clip_excerpt(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def extract_excerpt(text: str, config: CgvfLiteConfig) -> str:
    if not text:
        return ""
    lowered = text.lower()
    match = re.search(r"(explanation|reason)\s*:", lowered)
    if match:
        snippet = text[match.end() :].strip()
        sentences = _split_sentences(snippet)
        selected = sentences[: config.max_sentences]
        if len(selected) < config.min_sentences:
            selected = sentences
        excerpt = " ".join(selected)
        return _clip_excerpt(excerpt, config.max_excerpt_chars)

    lines = [line for line in text.strip().splitlines() if line.strip()]
    content = "\n".join(lines[:-1]) if len(lines) > 1 else text
    sentences = _split_sentences(content)
    if sentences:
        take_n = min(config.max_sentences, len(sentences))
        excerpt = " ".join(sentences[-take_n:])
        return _clip_excerpt(excerpt, config.max_excerpt_chars)

    fallback = _clip_excerpt(content, config.max_excerpt_chars)
    return fallback


def _select_excerpts(branches: Sequence[Dict[str, Any]], config: CgvfLiteConfig) -> List[str]:
    excerpts = [extract_excerpt(branch.get("model_output", ""), config) for branch in branches]
    non_empty = [excerpt for excerpt in excerpts if excerpt]
    non_empty.sort(key=len, reverse=True)
    return non_empty[: config.max_excerpts_per_group]


def rewrite_to_hypothesis(text: str, pred: str, alt: str) -> str:
    if not text:
        return text
    rewritten = text
    substitutions = [
        (r"the correct answer is\s+([ABCD])", r"a plausible hypothesis is \1"),
        (r"correct answer:\s*([ABCD])", r"hypothesis: \1"),
        (r"answer:\s*([ABCD])", r"hypothesis: \1"),
        (r"correct option:\s*([ABCD])", r"most likely option: \1"),
    ]
    for pattern, replacement in substitutions:
        rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
    rewritten = rewritten.rstrip()
    rewritten = f"{rewritten}\nAlternative: {alt} is also plausible; re-check the key discriminator."
    return rewritten


def tokenize_option(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.split(r"[^a-z0-9]+", text.lower())
    return [token for token in tokens if len(token) >= 6 and token not in _STOPWORDS]


def _mentions_unique(excerpts: Sequence[str], unique_tokens: Sequence[str]) -> bool:
    if not unique_tokens:
        return False
    for excerpt in excerpts:
        lower = excerpt.lower()
        if any(token in lower for token in unique_tokens):
            return True
    return False


def _build_rewritten_summary(
    counts: Dict[str, int],
    n_valid: int,
    top1: str,
    top2: str,
    excerpts_top1: Sequence[str],
    excerpts_top2: Sequence[str],
) -> str:
    lines = [
        f"Vote counts: A={counts['A']}, B={counts['B']}, C={counts['C']}, D={counts['D']}; N_valid={n_valid}.",
    ]
    for excerpt in excerpts_top1:
        lines.append(f"Branch supporting {top1}: {excerpt}")
    for excerpt in excerpts_top2:
        lines.append(f"Branch supporting {top2}: {excerpt}")
    return "\n".join(lines)


def parse_verify_response(text: str) -> Optional[VerifyResult]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    is_correct = payload.get("is_correct")
    final = payload.get("final")
    reason = payload.get("reason")
    if not isinstance(is_correct, bool):
        return None
    if not isinstance(final, str) or final not in VALID_CHOICES:
        return None
    if not isinstance(reason, str):
        return None
    return VerifyResult(is_correct=is_correct, final=final, reason=reason)


def run_cgvf_lite(
    *,
    question: str,
    options: Dict[str, str],
    branches: Sequence[Dict[str, Any]],
    verify_fn: Verifier,
    config: Optional[CgvfLiteConfig] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run CGVF-lite controller and return final prediction + log record."""
    config = config or CgvfLiteConfig()
    counts, n_valid = compute_vote_counts(branches)
    ranked = rank_labels(counts)
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else None
    top3 = ranked[2] if len(ranked) > 2 else None

    consensus = counts[top1] / n_valid if n_valid else 0.0
    gap = counts[top1] - counts[top2] if top2 else counts[top1]
    low = _low_consensus(consensus, gap, config)
    eligible = _eligible_vote_pattern(counts, top1, top2, top3, config)

    vf_records: Dict[str, Optional[Dict[str, Any]]] = {"r1": None, "r2": None, "r1b": None, "r2b": None}
    vf_conflict = False
    vf_floats = False

    raw_excerpts_top1 = _select_excerpts([b for b in branches if b.get("pred") == top1], config)
    raw_excerpts_top2 = _select_excerpts([b for b in branches if b.get("pred") == top2], config)
    excerpts_top1 = [
        rewrite_to_hypothesis(excerpt, top1, top2 or top1) for excerpt in raw_excerpts_top1
    ]
    excerpts_top2 = [
        rewrite_to_hypothesis(excerpt, top2 or top1, top1) for excerpt in raw_excerpts_top2
    ]

    rewritten_summary = _build_rewritten_summary(
        counts,
        n_valid,
        top1,
        top2 or top1,
        excerpts_top1,
        excerpts_top2,
    )

    tokens_top1 = tokenize_option(options.get(top1, ""))
    tokens_top2 = tokenize_option(options.get(top2, "")) if top2 else []
    unique_top1 = [tok for tok in tokens_top1 if tok not in tokens_top2]
    unique_top2 = [tok for tok in tokens_top2 if tok not in tokens_top1]
    mentions_top1 = _mentions_unique(raw_excerpts_top1, unique_top1)
    mentions_top2 = _mentions_unique(raw_excerpts_top2, unique_top2)
    lex_support_top2 = (not mentions_top1) and mentions_top2

    if not low or not eligible or top2 is None:
        log_record = {
            "top1": top1,
            "top2": top2,
            "counts": counts,
            "N_valid": n_valid,
            "consensus": consensus,
            "gap": gap,
            "low_consensus": low,
            "lex_support_top2": lex_support_top2,
            "vf": vf_records,
            "vf_conflict": vf_conflict,
            "vf_floats": vf_floats,
            "final_pred": top1,
            "flipped": False,
        }
        return top1, log_record

    r1 = verify_fn(top1, 1, rewritten_summary)
    r2 = verify_fn(top2, 1, rewritten_summary)
    vf_records["r1"] = asdict(r1) if r1 else None
    vf_records["r2"] = asdict(r2) if r2 else None

    vf_conflict = (
        r1 is None
        or r2 is None
        or r1.final == top2
        or r2.final == top1
        or r1.final not in {top1, top2}
        or r2.final not in {top1, top2}
    )

    r1b = None
    r2b = None
    if vf_conflict:
        r1b = verify_fn(top1, 2, rewritten_summary)
        r2b = verify_fn(top2, 2, rewritten_summary)
        vf_records["r1b"] = asdict(r1b) if r1b else None
        vf_records["r2b"] = asdict(r2b) if r2b else None

    def _contradiction(result: Optional[VerifyResult], candidate: str) -> bool:
        if result is None:
            return False
        if result.is_correct and result.final != candidate:
            return True
        if (not result.is_correct) and result.final == candidate:
            return True
        return False

    if vf_conflict:
        vf_floats = (
            r1 is None
            or r2 is None
            or r1b is None
            or r2b is None
            or (r1b is not None and r1 is not None and r1b.final != r1.final)
            or (r2b is not None and r2 is not None and r2b.final != r2.final)
            or _contradiction(r1, top1)
            or _contradiction(r2, top2)
            or _contradiction(r1b, top1)
            or _contradiction(r2b, top2)
        )
    else:
        vf_floats = False

    support_top2 = (
        r1 is not None
        and r2 is not None
        and r1.is_correct is False
        and r1.final == top2
        and r2.is_correct is True
        and r2.final == top2
        and (r1b is None or r1b.final == top2)
        and (r2b is None or r2b.final == top2)
        and not vf_floats
    )

    final_pred = top2 if support_top2 else top1
    log_record = {
        "top1": top1,
        "top2": top2,
        "counts": counts,
        "N_valid": n_valid,
        "consensus": consensus,
        "gap": gap,
        "low_consensus": low,
        "lex_support_top2": lex_support_top2,
        "vf": vf_records,
        "vf_conflict": vf_conflict,
        "vf_floats": vf_floats,
        "final_pred": final_pred,
        "flipped": final_pred == top2,
    }
    return final_pred, log_record
