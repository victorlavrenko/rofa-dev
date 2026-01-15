"""Helpers for resolving run directories."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from .io import load_manifest
from .model_id import to_slug


def ensure_model_slug_in_path(base_dir: str, model_slug: str) -> str:
    """Ensure the model slug is present as a path component in base_dir."""
    path = Path(base_dir)
    if model_slug in path.parts:
        return str(path)
    return str(path / model_slug)


def infer_model_slug(model_id: str, model_slug: Optional[str] = None) -> str:
    """Return a stable model slug, computing it when missing."""
    return model_slug or to_slug(model_id)


def _manifest_created_at(manifest_path: Path) -> datetime:
    manifest = load_manifest(str(manifest_path))
    if manifest and manifest.created_at:
        try:
            return datetime.fromisoformat(manifest.created_at)
        except ValueError:
            pass
    return datetime.fromtimestamp(manifest_path.stat().st_mtime)


def find_latest_run_dir(
    runs_root: str, model_slug: Optional[str], *, method: Optional[str] = None
) -> str:
    """Find the latest run directory for a model slug (optionally filtered by method)."""
    root = Path(runs_root)
    if not root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    candidates: list[tuple[datetime, Path]] = []
    for manifest_path in root.rglob("manifest.json"):
        if model_slug and model_slug not in manifest_path.parts:
            continue
        if method:
            manifest = load_manifest(str(manifest_path))
            if manifest is None or manifest.method != method:
                continue
        candidates.append((_manifest_created_at(manifest_path), manifest_path.parent))

    if not candidates:
        if model_slug:
            raise FileNotFoundError(
                f"No run directories found for model slug '{model_slug}' under {runs_root}."
            )
        raise FileNotFoundError(f"No run directories found under {runs_root}.")

    candidates.sort(key=lambda item: item[0], reverse=True)
    return str(candidates[0][1])
