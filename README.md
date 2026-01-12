# ROFA — Research-Oriented Framework for Advanced Reasoning

ROFA is an experimental research codebase aimed at studying **non-human-like reasoning regimes in large language models**, starting with controlled multi-branch sampling and consensus analysis, and designed to grow toward more advanced forms of reasoning (near-miss analysis, branch interaction, and spatial / 4D imagination).

This repository contains the code required to reproduce ROFA experiments across multiple papers, and currently has only one experiment.

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

## Papers and experiments

ROFA is now organized **by paper**, with shared infrastructure under `rofa/core/` and
paper-specific methods, prompts, and analyses under `rofa/papers/<paper_id>/`.

### Multi-paper architecture (umbrella repo)

- **Core toolkit**: `rofa/core/` (dataset selection, parsing, metrics, run I/O)
- **Paper packages**: `rofa/papers/<paper_slug>/` (prompts, methods, paper analysis glue)
- **Paper docs**: `docs/papers/<paper_slug>/`
- **Notebooks**: `notebooks/<paper_slug>/` (generation + reproduction)

### Choose your paper

- `from_answers_to_hypotheses`
  - Paper package: `rofa/papers/from_answers_to_hypotheses/`
  - Paper manuscript: `docs/papers/from_answers_to_hypotheses/paper.md`

The first paper captured in this repository is:

- **From Answers to Hypotheses: Parallel Clinical Reasoning as a Decision Paradigm for Medical AI**
  - Paper package: `rofa/papers/from_answers_to_hypotheses/`
  - Paper manuscript: `docs/papers/from_answers_to_hypotheses/paper.md`

## Scope of the baseline paper

The baseline paper currently supports **exactly two reasoning methods**:

1. **Greedy decoding**
   - Single deterministic generation
   - Standard baseline

2. **K-sample ensemble (10 branches; CLI alias: `branches`)**
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
   - Run `scripts/generate.py` or the Colab notebooks
   - Logs are written incrementally and safely

2. **Analyze logs** (local, CPU)
   - Run `scripts/analyze.py` or `notebooks/from_answers_to_hypotheses/20_paper_reproduce.ipynb`
   - No model access required

3. **Publish results**
   - Zip the run directory
   - Upload to GitHub Releases (manual)
   - Others can reproduce all metrics from logs alone

See `docs/REPRODUCIBILITY.md` for details.

## Reproduce paper results

- Notebook: `notebooks/from_answers_to_hypotheses/20_paper_reproduce.ipynb`
- Paper source: `docs/papers/from_answers_to_hypotheses/paper.tex`
- Paper PDF: `docs/papers/from_answers_to_hypotheses/paper.pdf`

## Non-Python quickstart (Colab + download)

1. Open `notebooks/from_answers_to_hypotheses/10_colab_generate.ipynb` in Colab and run the bootstrap cell.
2. Run the greedy and k-sample ensemble cells (two runs; same question set).
3. Download the run folders as zip files.
4. Open `notebooks/from_answers_to_hypotheses/20_paper_reproduce.ipynb`, paste local paths or zip paths, and run.

---

## Colab quickstart

### Run Greedy and k-sample ensemble on Colab

1. Open `notebooks/from_answers_to_hypotheses/10_colab_generate.ipynb` or launch it in Colab:
   - https://colab.research.google.com/github/victorlavrenko/rofa/blob/main/notebooks/from_answers_to_hypotheses/10_colab_generate.ipynb
2. Update `N`, `SEED`, or `OUT_BASE` if desired.
3. Run all cells. Outputs are written directly to Google Drive.

### Reproduce paper figures

1. Open `notebooks/from_answers_to_hypotheses/20_paper_reproduce.ipynb` or launch it in Colab:
   - https://colab.research.google.com/github/victorlavrenko/rofa/blob/main/notebooks/from_answers_to_hypotheses/20_paper_reproduce.ipynb
2. Provide a local run folder or a Release asset URL.
3. Run the analysis cells and export reports.

### Where outputs are saved

Runs are written under:

```
<out-dir>/<run-id>/
```

Each run contains:

- `summary.jsonl`
- `manifest.json`
- `progress.json`
- `question_set.json`
  - `full.jsonl` (k-sample ensemble only)

### How to resume after crash

Re-run the exact same command with the same `--out-dir` (and `--run-id` if used).
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
