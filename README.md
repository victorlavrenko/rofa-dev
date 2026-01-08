# ROFA — Research-Oriented Framework for Advanced Reasoning

ROFA is an experimental research codebase aimed at studying **non-human-like reasoning regimes in large language models**, starting with controlled multi-branch sampling and consensus analysis, and designed to grow toward more advanced forms of reasoning (near-miss analysis, branch interaction, and spatial / 4D imagination).

This repository contains the **code required to reproduce the first ROFA experiment**, as described in the accompanying research note, and is intentionally structured to look and behave like a **serious, reproducible ML research project**, not a one-off notebook.

---

## What problem ROFA is exploring

Most LLM evaluations treat reasoning as a **single linear process**:
> one prompt → one chain of thought → one answer.

ROFA explores a different hypothesis:

> **Reasoning can be treated as a *distribution* over parallel hypotheses**, not a single trajectory.

Concretely, ROFA studies what happens when:
- the *same prompt* is sampled multiple times,
- each sample represents a different reasoning path,
- and we analyze the *structure* of agreement, disagreement, and failure modes across these paths.

This allows us to study phenomena that are invisible in standard greedy decoding:
- false confidence (wrong unanimous answers),
- near-misses (gold answer appears but loses majority),
- ambiguity regimes (high entropy, low consensus),
- limits of majority voting as an accuracy booster.

---

## Scope of the current repository

This repository supports **exactly two reasoning methods**:

1. **Greedy decoding**
   - Single deterministic generation
   - Standard baseline

2. **Branch sampling ensemble (10 branches)**
   - 10 independent sampled generations
   - Same prompt, same question
   - Aggregation and consensus analysis over answers

> ⚠️ Important:  
> The prompt, reasoning request, decoding logic, parsing logic, and metric formulas are **not experimental variables** here.  
> They are preserved *exactly* as in the original notebook to ensure scientific continuity.

Future ROFA experiments (near-miss re-querying, branch interaction, spatial reasoning) are **out of scope for this repository**, but the code structure is explicitly designed to support them without refactoring the core.

---

## Relationship to the paper / research note

This codebase is intended to **support and reproduce the results described in the ROFA baseline experiment**, including:

- comparison of greedy vs multi-branch accuracy,
- analysis of consensus strength (`max_frac`) vs correctness,
- identification of unanimous but wrong cases,
- demonstration that majority voting does *not* reliably improve accuracy,
- empirical motivation for richer reasoning architectures.

The repository is organized so that:
- **generation** (GPU, Colab-friendly) and
- **analysis** (CPU-only, local)

are cleanly separated, enabling others to reuse published logs without re-running models.

---

## Metrics computed by this code

The following metrics are computed **exactly as defined in the research note** and must not be reinterpreted.

### Per-example (branch method only)

Given 10 branch predictions:

- **valid_n**  
  Number of branches that produced a valid answer (A/B/C/D)

- **none_n**  
  Number of branches that failed to produce a valid answer

- **leader**  
  Most frequent predicted answer (tie-breaking follows original logic)

- **leader_correct**  
  Whether `leader == gold`

- **max_frac**  
  Fraction of valid predictions supporting the leader

- **variation_ratio**  
  `1 − max_frac`

- **entropy_bits**  
  Shannon entropy (base-2) over answer distribution

### Aggregate / analysis metrics

- **Greedy accuracy**
- **Leader accuracy**
- **Unanimous cases** (`max_frac == 1.0`)
  - count
  - accuracy
- **Near-unanimous cases** (`max_frac ≥ 0.9`)
  - count
  - accuracy
- **Top-2 coverage**
  - gold answer appears among two most frequent predictions
- **R / W / Other classification**
  - Robust correct consensus
  - Wrong consensus
  - Low-consensus / ambiguous cases

These metrics are intentionally simple but expose **structural failure modes** that single-trajectory evaluation hides.

---

## Repository philosophy

This repository follows several explicit design principles:

- **No hidden logic in notebooks**  
  Notebooks are thin wrappers around library and CLI code.

- **Colab-safe generation**  
  Logs are written incrementally with resume support because Colab sessions can terminate without warning.

- **Self-describing artifacts**  
  Every run produces:
  - `summary.jsonl` — per-example results
  - `manifest.json` — configuration and environment
  - `progress.json` — resume state

- **Release-friendly**  
  Generated runs are designed to be published as GitHub Release assets and reused by others.

- **Research over engineering cleverness**  
  The goal is clarity, reproducibility, and analytical power — not maximum abstraction or framework complexity.

---

## What this repository is *not*

- It is **not** a benchmark leaderboard.
- It is **not** an optimized inference framework.
- It is **not** proposing JSON-only or constrained decoding.
- It is **not** claiming that majority voting is sufficient.

ROFA is an exploration tool, not a product.

---

## How to use this repository (high level)

1. **Generate logs** (GPU / Colab)
   - Run `scripts/generate.py` or the Colab notebook
   - Logs are written incrementally and safely

2. **Analyze logs** (local, CPU)
   - Run `scripts/analyze.py` or the analysis notebook
   - No model access required

3. **Publish results**
   - Zip the run directory
   - Upload to GitHub Releases
   - Others can reproduce all metrics from logs alone

See `docs/REPRODUCIBILITY.md` for details.

---

## Colab quickstart

### Run Greedy on Colab

1. Open `notebooks/01_colab_generate_greedy.ipynb`.
2. Update `N`, `SEED`, or `OUT_BASE` if desired.
3. Run all cells. Outputs are written directly to Google Drive.

### Run Branches on Colab

1. Open `notebooks/02_colab_generate_branches.ipynb`.
2. Update `N`, `SEED`, `BRANCHES`, or `OUT_BASE` if desired.
3. Run all cells. Outputs are written directly to Google Drive.

### Where outputs are saved

Runs are written under:

```
<out-dir>/<run-id>/
```

Each run contains:

- `summary.jsonl`
- `manifest.json`
- `progress.json`

### How to resume after crash

Re-run the exact same command with the same `--out-dir` and `--run-id`.
The generator detects `progress.json` and continues without duplicating records.

### How to upload a run folder to GitHub Releases

1. Zip the run directory:
   ```bash
   python scripts/pack_run.py --run-dir /path/to/<run-id>
   ```
2. Upload the resulting `.zip` to a GitHub Release.
3. Reviewers can validate the artifacts locally:
   ```bash
   python scripts/validate_run.py --run-dir /path/to/<run-id>
   ```

---

## Status

This repository represents **ROFA – Baseline Experiment (v1)**.

Future work (not implemented here):
- near-miss re-querying
- branch-to-branch interaction
- structured spatial / temporal representations
- non-textual hypothesis spaces

Those extensions are intentionally deferred until the baseline behavior is fully understood and documented.

---

## License / usage

This repository is provided for research and analysis purposes.  
If you reuse the code or logs, please preserve attribution and clearly state which ROFA experiment version you are reproducing.
