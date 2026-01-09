# ROFA Reproducibility Guide

This guide explains how to generate and analyze ROFA runs in a reproducible way.

## Generation (GPU required)

Generation requires a GPU. Colab is supported and recommended for most users.

### Colab notebooks

1. Open `notebooks/10_colab_generate.ipynb`.
2. Install dependencies, mount Google Drive, and set your output base path.
3. Choose the method cell (greedy or branches) and run the CLI command to start generation.

The generator writes `summary.jsonl` incrementally and saves `progress.json` after
each example, making it safe to resume after Colab restarts. Re-running the same
command with the same `--out-dir` (and `--run-id` if used) continues from the next example.

### CLI example

```bash
python scripts/generate.py \\
  --method branches \\
  --n 200 \\
  --seed 42 \\
  --branches 10 \\
  --temperature 0.8 \\
  --out-dir /content/drive/MyDrive/rofa_runs/example_run
```

## Analysis (CPU-only)

Analysis runs locally without a GPU as long as you have the saved run folder.

1. Download or mount the run directory created in Colab.
2. Run `notebooks/20_paper_reproduce.ipynb` (local or Colab) to compute paper metrics.

Example (CLI alternative):

```bash
python scripts/analyze.py --summary /path/to/run/summary.jsonl
```

A machine-readable report is written alongside the summary log as `report.json`.

## Publishing to GitHub Releases

1. Zip the run directory (it contains all required artifacts).
2. Upload the archive to a GitHub Release.
3. Copy the asset URL into `notebooks/20_paper_reproduce.ipynb`.
4. Consumers can re-run analysis without access to the model weights.

## End-to-end workflow

1. Generate a run in Colab (greedy or branches).
2. Upload the run zip to GitHub Releases (manual).
3. Reproduce tables and metrics in `notebooks/20_paper_reproduce.ipynb`.

## Reproducing headline numbers

- Use the same `summary.jsonl` logs that were produced by `scripts/generate.py`.
- Run `scripts/analyze.py` to recompute greedy accuracy, leader accuracy, and
  consensus statistics.
- Verify the output report against published tables.
