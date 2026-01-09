#!/usr/bin/env bash
set -euo pipefail

python -m pyright .
python -m ruff check .
python -m pytest
