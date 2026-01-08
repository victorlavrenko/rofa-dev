"""Shared dataset loop for generation runs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

from datasets import load_dataset

from .io import _append_jsonl, _now_utc, load_progress, write_progress
from .parse import cop_to_letter


def load_filtered_dataset(dataset_name: str, split: str):
    """Load and filter the dataset with the same criteria as the notebook."""
    ds = load_dataset(dataset_name, split=split)
    ds = ds.filter(
        lambda x: (
            x.get("choice_type") == "single"
            and isinstance(x.get("exp"), str)
            and len(x["exp"]) > 20
            and len(x["exp"]) < 500
        )
    )
    return ds


def run_dataset_loop(
    *,
    method,
    output_summary_path: str,
    output_full_path: Optional[str],
    progress_path: str,
    dataset_name: str,
    dataset_split: str,
    seed: int,
    n: int,
    subjects: int,
    max_new_tokens: int,
    tokenizer,
    model,
    run_id: str,
) -> Dict[str, Any]:
    """Run the dataset loop with subject balancing and resume support."""
    max_per_subject = n / subjects * 1.1 + 1

    ds = load_filtered_dataset(dataset_name, dataset_split)
    shuffled = ds.shuffle(seed=seed)

    progress = load_progress(progress_path)
    if progress:
        i = progress.get("i", 0)
        picked = progress.get("picked", 0)
        subject_counts = Counter(progress.get("subject_counts", {}))
        summary_written = progress.get("summary_written", 0)
        full_written = progress.get("full_written", 0)
    else:
        i = 0
        picked = 0
        subject_counts = Counter()
        summary_written = 0
        full_written = 0
        open(output_summary_path, "w", encoding="utf-8").close()
        if output_full_path:
            open(output_full_path, "w", encoding="utf-8").close()

    while picked < n and i < len(shuffled):
        ex = shuffled[i]
        subj = ex.get("subject_name", "Unknown") or "Unknown"

        if subject_counts[subj] >= max_per_subject:
            i += 1
            write_progress(
                progress_path,
                {
                    "run_id": run_id,
                    "timestamp": _now_utc(),
                    "i": i,
                    "picked": picked,
                    "subject_counts": dict(subject_counts),
                    "summary_written": summary_written,
                    "full_written": full_written,
                },
            )
            continue

        subject_counts[subj] += 1
        picked += 1

        context = {
            "tokenizer": tokenizer,
            "model": model,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "index": i,
            "picked_index": picked,
            "subject_name": subj,
            "gold": cop_to_letter(ex["cop"]),
        }

        record = method.run_one(ex, context)
        _append_jsonl(output_summary_path, record)
        summary_written += 1

        full_record = getattr(method, "last_full_record", None)
        if output_full_path and full_record is not None:
            _append_jsonl(output_full_path, full_record)
            full_written += 1

        if record.get("prediction") is None and "prediction" in record:
            print("  Warning: could not extract answer from model output.")
            picked -= 1

        i += 1

        write_progress(
            progress_path,
            {
                "run_id": run_id,
                "timestamp": _now_utc(),
                "i": i,
                "picked": picked,
                "subject_counts": dict(subject_counts),
                "summary_written": summary_written,
                "full_written": full_written,
            },
        )

    return {
        "summary_written": summary_written,
        "full_written": full_written,
        "picked": picked,
        "i": i,
    }
