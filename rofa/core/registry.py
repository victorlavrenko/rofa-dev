"""Registries for papers and methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Protocol


class Method(Protocol):
    """Protocol for method implementations used by the core runner."""

    def run_one(self, example: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the method on a single example and return a summary record."""


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
_METHOD_ALIASES: Dict[str, str] = {
    "branches": "k_sample_ensemble",
}


def register_paper(spec: PaperSpec) -> PaperSpec:
    """Register a paper and return the stored spec.

    Papers should expose a ``PaperSpec`` containing explicit ``MethodSpec`` entries.
    Use :func:`resolve_method_key` to keep backward-compatible aliases such as
    ``branches`` mapped to the internal method name ``k_sample_ensemble``.
    """
    if spec.paper_id in _PAPERS:
        raise ValueError(f"Paper '{spec.paper_id}' is already registered.")
    _PAPERS[spec.paper_id] = spec
    return spec


def get_paper(paper_id: str) -> PaperSpec:
    """Fetch a registered paper spec by paper_id."""
    if paper_id not in _PAPERS:
        raise KeyError(f"Paper '{paper_id}' is not registered.")
    return _PAPERS[paper_id]


def list_papers() -> Dict[str, PaperSpec]:
    """Return all registered paper specs."""
    return dict(_PAPERS)


def resolve_method_key(method: str) -> str:
    """Resolve backward-compatible method aliases to internal names."""
    return _METHOD_ALIASES.get(method, method)


def list_method_aliases() -> Dict[str, str]:
    """Return the supported method aliases for CLI compatibility."""
    return dict(_METHOD_ALIASES)
