"""Question set utilities for deterministic ROFA generation."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from .io import _atomic_write_json


@dataclass
class QuestionSet:
    """Serializable question set definition."""

    qs_id: str
    dataset_name: str
    dataset_split: str
    dataset_revision: Optional[str]
    dataset_fingerprint: Optional[str]
    selection: Dict[str, Any]
    examples: List[Dict[str, Any]]


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


def question_hash(example: Dict[str, Any]) -> str:
    payload = {
        "question": example.get("question"),
        "opa": example.get("opa"),
        "opb": example.get("opb"),
        "opc": example.get("opc"),
        "opd": example.get("opd"),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _make_qs_id(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def create_question_set(dataset_cfg: Dict[str, Any], selection_cfg: Dict[str, Any]) -> QuestionSet:
    """Create a deterministic question set."""
    dataset_name = dataset_cfg["dataset_name"]
    dataset_split = dataset_cfg["dataset_split"]

    ds = load_filtered_dataset(dataset_name, dataset_split)
    indices = list(range(len(ds)))
    random.Random(selection_cfg["seed"]).shuffle(indices)

    max_per_subject = selection_cfg["max_per_subject"]
    subject_counts: Counter[str] = Counter()
    examples: List[Dict[str, Any]] = []

    for idx in indices:
        ex = ds[idx]
        subj = ex.get("subject_name", "Unknown") or "Unknown"
        if subject_counts[subj] >= max_per_subject:
            continue
        subject_counts[subj] += 1

        examples.append(
            {
                "dataset_index": idx,
                "id": ex.get("id"),
        "question_hash": question_hash(ex),
            }
        )

        if len(examples) >= selection_cfg["n"]:
            break

    selection = {
        "seed": selection_cfg["seed"],
        "n": selection_cfg["n"],
        "subjects": selection_cfg["subjects"],
        "max_per_subject": max_per_subject,
        "filters": {
            "choice_type": "single",
            "exp_len_min": 20,
            "exp_len_max": 500,
        },
        "subject_counts": dict(subject_counts),
    }

    qs_payload = {
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "selection": selection,
        "examples": examples,
    }
    qs_id = _make_qs_id(qs_payload)

    return QuestionSet(
        qs_id=qs_id,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_revision=str(getattr(ds.info, "version", None))
        if ds.info is not None
        else None,
        dataset_fingerprint=getattr(ds, "_fingerprint", None),
        selection=selection,
        examples=examples,
    )


def save_question_set(qs: QuestionSet, path: str) -> None:
    """Save a question set to disk."""
    _atomic_write_json(path, asdict(qs))


def load_question_set(path: str) -> QuestionSet:
    """Load a question set from disk."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return QuestionSet(**payload)
