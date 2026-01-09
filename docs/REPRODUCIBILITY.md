# ROFA Reproducibility Guide

This guide provides step-by-step workflows for generating, validating, packaging, and analyzing
ROFA runs. It supports both Colab (GPU) and local CPU analysis.

## Colab route (recommended for generation)

1. Open `notebooks/10_colab_generate.ipynb` in Colab.
2. Run the bootstrap cell (mount Drive, clone repo, install).
3. Run the **greedy** cell and the **k-sample ensemble** cell (branches alias).
4. Note the run directories created under your Drive (e.g., `.../rofa_runs/runs/<run_id>`).

The generator writes `summary.jsonl` incrementally and saves `progress.json` after each example,
so re-running the same command with the same `--out-dir` / `--run-id` will resume safely.

## Local route (CPU-only analysis)

1. Copy the run directory from Colab to your machine.
2. Open `notebooks/20_paper_reproduce.ipynb` and set the run paths.
3. Run the notebook to compute paper metrics.

CLI alternative:

```bash
python scripts/analyze.py --run /path/to/run_dir
```

Reports are written under `notebooks/reports/<run_id>/report.json` by default.

## Validate a run (recommended before analysis)

```bash
python scripts/validate_run.py --run /path/to/run_dir
```

The validator checks:
- required artifacts exist (`manifest.json`, `summary.jsonl`, `progress.json`)
- JSONL parses cleanly
- required columns are present for the method

## Pack a run for sharing

```bash
python scripts/pack_run.py --run-dir /path/to/run_dir
```

This creates `/path/to/run_dir.zip` that can be uploaded or shared.

## Publish via GitHub Releases (manual, no auth required)

1. Zip the run directory using `scripts/pack_run.py`.
2. Create a GitHub Release in the repository UI.
3. Upload the zip as a release asset.
4. Copy the asset URL into `notebooks/20_paper_reproduce.ipynb` (asset URL field).

## End-to-end workflow checklist

1. Generate runs in Colab (greedy + k-sample ensemble).
2. Validate the runs with `scripts/validate_run.py`.
3. Pack the runs with `scripts/pack_run.py`.
4. Analyze locally with `notebooks/20_paper_reproduce.ipynb` or `scripts/analyze.py`.
