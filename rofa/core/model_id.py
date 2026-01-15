"""Model identification helpers."""

from __future__ import annotations

import re


def to_slug(model_id: str) -> str:
    """Convert a Hugging Face model id to a filesystem-safe slug."""
    if not isinstance(model_id, str):
        raise TypeError("model_id must be a string.")
    slug = model_id.strip().lower()
    slug = slug.replace("/", "__")
    slug = re.sub(r"[^a-z0-9._-]", "_", slug)
    return slug
