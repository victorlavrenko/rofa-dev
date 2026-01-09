"""I/O utilities for run artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .schemas import RunManifest


def _now_utc() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """Append a JSONL record with flush + fsync for durability."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write JSON atomically by replacing a temp file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dir_name = os.path.dirname(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dir_name, delete=False) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def write_manifest(path: str, manifest: RunManifest) -> None:
    """Write the run manifest to disk.

    Args:
        path: Destination path for ``manifest.json``.
        manifest: RunManifest payload.

    Artifacts:
        Writes a JSON manifest file containing stable run metadata.
    """
    payload = asdict(manifest)
    _atomic_write_json(path, payload)


def write_progress(path: str, payload: Dict[str, Any]) -> None:
    """Persist progress metadata for resume support.

    Args:
        path: Destination path for ``progress.json``.
        payload: Progress metadata including ``run_id`` and record counters.

    Artifacts:
        Writes a JSON progress file used for resume logic.
    """
    _atomic_write_json(path, payload)


def load_progress(path: str) -> Optional[Dict[str, Any]]:
    """Load progress metadata if it exists.

    Returns:
        The progress payload or None if ``path`` does not exist.
    """
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def download(url: str, dest: str) -> str:
    """Download a file from a URL to the destination path.

    Args:
        url: Remote URL to fetch.
        dest: Local destination path.

    Returns:
        The destination path after writing.

    Raises:
        URLError: If the request fails or the remote file is unavailable.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f)
    return dest


def unpack_zip(zip_path: str, dst_dir: str) -> str:
    """Unpack a zip file into a destination directory."""
    os.makedirs(dst_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)
    return dst_dir
