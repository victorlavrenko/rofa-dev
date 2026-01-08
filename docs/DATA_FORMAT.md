# ROFA Data Format

This document describes the run artifact layout produced by `scripts/generate.py`.

## Run directory structure

Each run directory contains:

- `summary.jsonl` — required, per-example records for the selected method.
- `full.jsonl` — branch method only; contains full per-branch outputs.
- `manifest.json` — run metadata and configuration.
- `progress.json` — resume state for interrupted runs.

## `summary.jsonl`

### Greedy method records

Each line is a JSON object with the fields:

- `index`: integer index in the shuffled dataset.
- `id`: dataset example id.
- `question`: question text.
- `options`: map of `A`..`D` to option text.
- `gold`: gold answer letter.
- `prediction`: extracted answer letter or `null` if none.
- `is_correct`: `true`/`false`/`null` (null when prediction is missing).
- `model_output`: raw model generation text.
- `subject_name`: subject string.
- `inference_time_sec`: wall-time for generation.
- `model`: model identifier.
- `max_new_tokens`: generation cap.
- `seed`: base seed.
- `timestamp`: ISO UTC timestamp.

### Branch sampling records

Each line is a JSON object with the fields:

- `index`, `picked_index`, `id`, `gold`, `subject_name`, `timestamp`.
- `branch_preds`: list of predictions for each branch.
- `leader`, `max_frac`, `valid_n`, `none_n`, `variation_ratio`, `entropy_bits`.
- `correct_fraction`, `leader_correct`.
- `class`: categorical label used in the notebook (`unanimous`, `lead80`, `lead50`, `no_leader`, `invalid_all_none`).

## `full.jsonl` (branch method only)

Each line is a JSON object with:

- Example metadata (index, id, question, options, gold, subject).
- `branches`: per-branch records (seed, temperature, top_p, top_k, prediction, output).
- `branch_preds`: list of predictions.
- `metrics`: per-example metrics, including `class` and timing info.

## `manifest.json`

Contains:

- `run_id`: unique identifier.
- `created_at`: ISO UTC timestamp.
- `method`: `greedy` or `branches`.
- `config`: the run configuration (dataset, seeds, generation parameters).

## `progress.json`

Contains resume state:

- `run_id`: identifier matching the manifest.
- `i`: current dataset index.
- `picked`: number of accepted examples.
- `subject_counts`: subject histogram.
- `summary_written`: number of summary records written.
- `full_written`: number of full records written.
- `timestamp`: ISO UTC timestamp.
