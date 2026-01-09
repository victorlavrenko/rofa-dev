#!/usr/bin/env bash
set -euo pipefail

echo "==> Running pyright"
pyright .

echo "==> Running ruff"
ruff check .

echo "==> Running pytest"
pytest
