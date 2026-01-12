# From Answers to Hypotheses — Paper documentation

## Paper overview

**From Answers to Hypotheses: Internal Consensus and Its Limits in Large Language Models** is the
baseline paper captured in this repository. It analyzes multi-branch sampling and consensus behavior for
clinical QA prompts, contrasting greedy decoding with a 10-branch ensemble.

- Paper package: [`rofa/papers/from_answers_to_hypotheses/`](../../../rofa/papers/from_answers_to_hypotheses/)
- Paper manuscript: [`docs/papers/from_answers_to_hypotheses/paper.md`](paper.md)
- Paper source: [`docs/papers/from_answers_to_hypotheses/paper.tex`](paper.tex)
- Paper PDF: [`docs/papers/from_answers_to_hypotheses/paper.pdf`](paper.pdf)

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

Future ROFA experiments (near-miss re-querying, branch interaction, spatial reasoning) are **out of scope for this paper**,
but the code structure is explicitly designed to support them without refactoring the core.

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

## Reproduce paper results

### Notebooks

| Purpose | Local notebook | Colab notebook |
| --- | --- | --- |
| Generate runs (GPU) | [`notebooks/from_answers_to_hypotheses/10_colab_generate.ipynb`](../../../notebooks/from_answers_to_hypotheses/10_colab_generate.ipynb) | [Open in Colab](https://colab.research.google.com/github/victorlavrenko/rofa/blob/main/notebooks/from_answers_to_hypotheses/10_colab_generate.ipynb) |
| Reproduce metrics (CPU) | [`notebooks/from_answers_to_hypotheses/20_paper_reproduce.ipynb`](../../../notebooks/from_answers_to_hypotheses/20_paper_reproduce.ipynb) | [Open in Colab](https://colab.research.google.com/github/victorlavrenko/rofa/blob/main/notebooks/from_answers_to_hypotheses/20_paper_reproduce.ipynb) |

### Paper artifacts

- Paper source: [`docs/papers/from_answers_to_hypotheses/paper.tex`](paper.tex)
- Paper PDF: [`docs/papers/from_answers_to_hypotheses/paper.pdf`](paper.pdf)
