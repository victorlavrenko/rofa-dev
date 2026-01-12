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

ROFA is organized **by paper**, with shared infrastructure under `rofa/core/` and
paper-specific methods, prompts, and analyses under `rofa/papers/<paper_id>/`.

### Multi-paper architecture (umbrella repo)

- **Core toolkit**: `rofa/core/` (dataset selection, parsing, metrics, run I/O)
- **Paper packages**: `rofa/papers/<paper_slug>/` (prompts, methods, paper analysis glue)
- **Paper docs**: `docs/papers/<paper_slug>/`
- **Notebooks**: `notebooks/<paper_slug>/` (generation + reproduction)

### Choose your paper

- `from_answers_to_hypotheses`
  - Paper package: `rofa/papers/from_answers_to_hypotheses/`
  - Paper documentation: `docs/papers/from_answers_to_hypotheses/README.md`

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

ROFA is an exploration tool, not yet a product.

---

## How to use this repository (high level)

1. **Generate logs** (GPU / Colab)
   - Run `scripts/generate.py` or the Colab notebooks
   - Logs are written incrementally and safely

2. **Analyze logs** (local, CPU)
   - Run `scripts/analyze.py` or `notebooks/<paper_slug>/20_paper_reproduce.ipynb`
   - No model access required

3. **Publish results**
   - Zip the run directory
   - Upload to GitHub Releases (manual)
   - Others can reproduce all metrics from logs alone

See `docs/REPRODUCIBILITY.md` for details.

## Reproduce paper results

- [`from_answers_to_hypotheses`](docs/papers/from_answers_to_hypotheses/README.md)
