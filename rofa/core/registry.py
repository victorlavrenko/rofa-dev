"""Registries for papers and methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Protocol


class Method(Protocol):
    """Protocol for method implementations."""

    def run_one(self, example, context):
        """Run the method on a single example."""


@dataclass(frozen=True)
class MethodSpec:
    """Metadata for a method implementation."""

    key: str
    factory: Callable[[], Method]
    description: str


@dataclass(frozen=True)
class PaperSpec:
    """Metadata for a paper and its available methods."""

    paper_id: str
    title: str
    description: str
    default_dataset: str
    default_split: str
    methods: Dict[str, MethodSpec]


_PAPERS: Dict[str, PaperSpec] = {}


def register_paper(spec: PaperSpec) -> PaperSpec:
    """Register a paper and return the stored spec."""
    if spec.paper_id in _PAPERS:
        raise ValueError(f"Paper '{spec.paper_id}' is already registered.")
    _PAPERS[spec.paper_id] = spec
    return spec


def get_paper(paper_id: str) -> PaperSpec:
    """Fetch a registered paper spec."""
    if paper_id not in _PAPERS:
        raise KeyError(f"Paper '{paper_id}' is not registered.")
    return _PAPERS[paper_id]


def list_papers() -> Dict[str, PaperSpec]:
    """Return all registered paper specs."""
    return dict(_PAPERS)
