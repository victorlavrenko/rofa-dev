"""Question set utilities for deterministic ROFA generation."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    """Load and filter the dataset with the same criteria as the notebook.

    Args:
        dataset_name: Hugging Face dataset name.
        split: Dataset split to load (e.g., ``validation``).

    Returns:
        A filtered datasets.Dataset object.

    Raises:
        ValueError: If the dataset cannot be loaded or filtering fails due to schema drift.
    """
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
    """Return a stable hash for the question/option fields."""
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


def _select_examples(ds, selection_cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Counter]:
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

    return examples, subject_counts


def create_question_set(dataset_cfg: Dict[str, Any], selection_cfg: Dict[str, Any]) -> QuestionSet:
    """Create a deterministic question set.

    Args:
        dataset_cfg: Dataset name and split configuration.
        selection_cfg: Selection settings (seed, n, subjects, max_per_subject).

    Returns:
        A QuestionSet with deterministic ordering and metadata.

    Artifacts:
        None (use :func:`save_question_set` to persist to disk).

    Raises:
        ValueError: If the dataset schema has drifted or selection criteria are invalid.

    Notes:
        The selection protocol (filters + subject balancing) is stable across papers;
        paper-specific analyses should treat the generated question set as a fixed input.
    """
    dataset_name = dataset_cfg["dataset_name"]
    dataset_split = dataset_cfg["dataset_split"]

    ds = load_filtered_dataset(dataset_name, dataset_split)
    examples, subject_counts = _select_examples(ds, selection_cfg)

    max_per_subject = selection_cfg["max_per_subject"]
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


def expand_question_set(
    qs: QuestionSet,
    dataset_cfg: Dict[str, Any],
    selection_cfg: Dict[str, Any],
) -> QuestionSet:
    """Expand a question set to a larger ``n`` without reshuffling earlier examples."""
    if selection_cfg["n"] <= len(qs.examples):
        return qs

    if qs.dataset_name != dataset_cfg["dataset_name"]:
        raise ValueError("Dataset name mismatch when expanding question set.")
    if qs.dataset_split != dataset_cfg["dataset_split"]:
        raise ValueError("Dataset split mismatch when expanding question set.")

    for key in ("seed", "subjects", "max_per_subject"):
        if qs.selection.get(key) != selection_cfg.get(key):
            raise ValueError(f"Selection mismatch on {key} when expanding question set.")

    ds = load_filtered_dataset(dataset_cfg["dataset_name"], dataset_cfg["dataset_split"])
    expanded_examples, subject_counts = _select_examples(ds, selection_cfg)
    existing_len = len(qs.examples)

    if len(expanded_examples) < existing_len:
        raise ValueError("Expanded question set is shorter than existing examples.")

    for old, new in zip(qs.examples, expanded_examples[:existing_len]):
        if (
            old.get("dataset_index") != new.get("dataset_index")
            or old.get("id") != new.get("id")
            or old.get("question_hash") != new.get("question_hash")
        ):
            raise ValueError("Existing question set does not match expanded selection.")

    selection = dict(qs.selection)
    selection.update(
        {
            "n": selection_cfg["n"],
            "subjects": selection_cfg["subjects"],
            "max_per_subject": selection_cfg["max_per_subject"],
            "subject_counts": dict(subject_counts),
        }
    )

    qs_payload = {
        "dataset_name": dataset_cfg["dataset_name"],
        "dataset_split": dataset_cfg["dataset_split"],
        "selection": selection,
        "examples": expanded_examples,
    }
    qs_id = _make_qs_id(qs_payload)

    return QuestionSet(
        qs_id=qs_id,
        dataset_name=dataset_cfg["dataset_name"],
        dataset_split=dataset_cfg["dataset_split"],
        dataset_revision=str(getattr(ds.info, "version", None))
        if ds.info is not None
        else None,
        dataset_fingerprint=getattr(ds, "_fingerprint", None),
        selection=selection,
        examples=expanded_examples,
    )


def save_question_set(qs: QuestionSet, path: str) -> None:
    """Save a question set to disk.

    Args:
        qs: QuestionSet instance.
        path: Destination file path for ``question_set.json``.

    Artifacts:
        Writes a JSON file at ``path``.
    """
    _atomic_write_json(path, asdict(qs))


def load_question_set(path: str) -> QuestionSet:
    """Load a question set from disk.

    Args:
        path: Path to ``question_set.json``.

    Returns:
        A QuestionSet instance.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the JSON payload is missing required keys.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return QuestionSet(**payload)
