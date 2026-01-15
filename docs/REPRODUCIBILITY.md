# ROFA Reproducibility Guide

This page is a lightweight index for paper-specific reproduction guides. Each paper documentation
includes the authoritative steps and notebook links.

## Single-model CLI generation

Run generation for a single Hugging Face model id (outputs are stored under a per-model slug):

```bash
HF_TOKEN=... python scripts/generate.py \
  --paper from_answers_to_hypotheses \
  --method branches \
  --model HPAI-BSC/Qwen2.5-Aloe-Beta-7B \
  --N 400 \
  --seed 42 \
  --k 10
```

## Reproduce from a run directory

```bash
python scripts/reproduce.py --run-dir runs/.../<model_slug>/.../
```

## Reproduce latest run for a model id

```bash
python scripts/reproduce.py --model HPAI-BSC/Qwen2.5-Aloe-Beta-7B
```

## Gated model access (401/403)

Some models (e.g., `google/medgemma-1.5-4b-it`) require accepting access terms on Hugging Face.

1. Open the model page in a browser while logged in and accept the terms.
2. Create a Hugging Face access token with read scope.
3. Provide the token via `HF_TOKEN` (or `--hf-token`).

If you still see 401/403 errors after providing a token, confirm that you accepted the model’s
access conditions on Hugging Face.

## Paper reproduction guides

- **From Answers to Hypotheses**: [`docs/papers/from_answers_to_hypotheses/README.md`](papers/from_answers_to_hypotheses/README.md)
