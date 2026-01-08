# ROFA Reproducibility Guide

This guide explains how to generate and analyze ROFA runs in a reproducible way.

## Generation (GPU required)

Generation requires a GPU. Colab is supported and recommended for most users.

### Colab notebooks

1. Open `notebooks/01_colab_generate_greedy.ipynb` or `notebooks/02_colab_generate_branches.ipynb`.
2. Install dependencies, mount Google Drive, and set your output base path.
3. Run the single CLI command cell to start generation.

The generator writes `summary.jsonl` incrementally and saves `progress.json` after
each example, making it safe to resume after Colab restarts. Re-running the same
command with the same `--out-dir` and `--run-id` continues from the next example.

### CLI example

```bash
python scripts/generate.py \
  --method branches \
  --n 200 \
  --seed 42 \
  --branches 10 \
  --temperature 0.8 \
  --out-dir /content/drive/MyDrive/rofa_runs \
  --run-id example_run
```

## Analysis (CPU-only)

Analysis runs locally without a GPU as long as you have the saved run folder.

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
