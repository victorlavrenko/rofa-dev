"""Schema definitions for run artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunConfig:
    """Configuration parameters for a run."""

    method: str
    model_id: str
    seed: int
    max_new_tokens: int
    n: int
    subjects: int
    max_per_subject: float
    dataset_name: str
    dataset_split: str
    n_branches: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    question_set_id: Optional[str] = None


@dataclass
class GenerationConfig:
    """Inputs required to run generation."""

    method: str
    model_id: str
    out_dir: str
    run_id: Optional[str] = None
    resume: Optional[bool] = None
    expand: bool = False
    seed: int = 42
    max_new_tokens: int = 1024
    n: int = 100
    subjects: int = 20
    dataset_name: str = "openlifescienceai/medmcqa"
    dataset_split: str = "validation"
    question_set_path: Optional[str] = None
    n_branches: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    progress: bool = False
    write_full_records: bool = False
    tokenizer: Optional[Any] = None
    model: Optional[Any] = None
    method_impl: Optional[Any] = None


@dataclass
class RunManifest:
    """Metadata for a run."""

    run_id: str
    created_at: str
    method: str
    config: RunConfig
    notes: Optional[str] = None


@dataclass
class LogRecord:
    """Generic log record (fields are optional for flexibility)."""

    index: int
    id: Optional[str]
    gold: Optional[str]
    subject_name: Optional[str]
    timestamp: Optional[str]
    data: Dict[str, Any] = field(default_factory=dict)
    branches: Optional[List[Dict[str, Any]]] = None
    branch_preds: Optional[List[Optional[str]]] = None
    leader: Optional[str] = None
    max_frac: Optional[float] = None
    valid_n: Optional[int] = None
    none_n: Optional[int] = None
    variation_ratio: Optional[float] = None
    entropy_bits: Optional[float] = None
    correct_fraction: Optional[float] = None
    leader_correct: Optional[bool] = None
    class_label: Optional[str] = None
