# ROFA Data Format

This document describes the run artifact layout produced by [`scripts/generate.py`](../scripts/generate.py) and
[`rofa/core/runner.py`](../rofa/core/runner.py). Field names match [`rofa/core/schemas.py`](../rofa/core/schemas.py).

## Run directory structure

Each run directory contains:

- `summary.jsonl` — required, per-example records for the selected method.
- `full.jsonl` — k-sample ensemble only; full per-branch outputs.
- `manifest.json` — run metadata and configuration.
- `progress.json` — resume state for interrupted runs.
- `question_set.json` — the sampled question set used for the run.

## `summary.jsonl`

### Greedy records

Each line is a JSON object with:

- `index`: integer index within the question set.
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

### K-sample ensemble records

Each line is a JSON object with:

- `index`, `picked_index`, `id`, `gold`, `subject_name`, `timestamp`.
- `branch_preds`: list of predictions for each branch.
- `leader`, `max_frac`, `valid_n`, `none_n`, `variation_ratio`, `entropy_bits`.
- `correct_fraction`, `leader_correct`.
- `class`: categorical label (`unanimous`, `lead80`, `lead50`, `no_leader`, `invalid_all_none`).

### Example `summary.jsonl` record (k-sample ensemble)

```json
{
  "index": 0,
  "picked_index": 1,
  "id": "medmcqa_123",
  "gold": "B",
  "branch_preds": ["B", "B", "C", "B", "A", "B", "B", "B", "B", "B"],
  "leader": "B",
  "max_frac": 0.8,
  "valid_n": 10,
  "none_n": 0,
  "variation_ratio": 0.19999999999999996,
  "entropy_bits": 0.7219,
  "correct_fraction": 0.9,
  "leader_correct": true,
  "class": "lead80",
  "subject_name": "Biology",
  "timestamp": "2026-01-08T13:28:26.123456+00:00"
}
```

## `full.jsonl` (k-sample ensemble only)

Each line is a JSON object with:

- Example metadata (index, id, question, options, gold, subject).
- `branches`: per-branch records (seed, temperature, top_p, top_k, pred, model_output).
- `branch_preds`: list of predictions.
- `metrics`: per-example metrics, including `class`, `correct_fraction`, `leader_correct`,
  `wall_time_sec`, and `mean_branch_time_sec`.

## `manifest.json`

Contains:

- `run_id`: unique identifier.
- `created_at`: ISO UTC timestamp.
- `method`: `greedy` or `k_sample_ensemble` (older runs may show `branches`).
- `config`: the run configuration:
  - `method`, `model_id`, `seed`, `max_new_tokens`, `n`, `subjects`, `max_per_subject`
  - `dataset_name`, `dataset_split`
  - `n_branches`, `branch_batch_size`, `temperature`, `top_p`, `top_k` (k-sample ensemble only)
  - `question_set_id`

## `progress.json`

Contains resume state:

- `run_id`: identifier matching the manifest.
- `position`: current dataset index.
- `summary_written`: number of summary records written.
- `full_written`: number of full records written.
- `timestamp`: ISO UTC timestamp.

## `question_set.json`

Contains:

- `qs_id`: question set identifier.
- `dataset_name`, `dataset_split`, `dataset_revision`, `dataset_fingerprint`
- `selection`: selection configuration (seed, n, subjects, max_per_subject, filters)
- `examples`: list of example references with `dataset_index`, `id`, `question_hash`
