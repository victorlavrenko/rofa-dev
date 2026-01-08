# ROFA Reproducibility Guide

This guide explains how to generate and analyze ROFA runs in a reproducible way.

## Colab generation

1. Open `notebooks/10_colab_generate.ipynb`.
2. Install dependencies and mount Google Drive.
3. Run the `scripts/generate.py` CLI with your output directory on Drive.

The generator writes `summary.jsonl` incrementally and saves `progress.json` after
 each example, making it safe to resume after Colab restarts.

## Local analysis

1. Download or mount the run directory created in Colab.
2. Run `scripts/analyze.py` against `summary.jsonl` to compute aggregate metrics.

Example:

```bash
python scripts/analyze.py --summary /path/to/run/summary.jsonl
```

A machine-readable report is written alongside the summary log as `report.json`.

## Publishing to GitHub Releases

1. Zip the run directory (it contains all required artifacts).
2. Upload the archive to a GitHub Release.
3. Consumers can re-run analysis without access to the model weights.

## Reproducing headline numbers

- Use the same `summary.jsonl` logs that were produced by `scripts/generate.py`.
- Run `scripts/analyze.py` to recompute greedy accuracy, leader accuracy, and
  consensus statistics.
- Verify the output report against published tables.
