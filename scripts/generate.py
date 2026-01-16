"""CLI entrypoint for generation runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from huggingface_hub import login

from rofa.core.model import (
    has_flash_attn,
    load_model_with_fallback,
    load_tokenizer,
    _resolve_attn_implementation,
)
from rofa.core.model_id import to_slug
from rofa.core.question_set import create_question_set, load_question_set, save_question_set
from rofa.core.registry import get_paper, list_method_aliases, resolve_method_key
from rofa.core.run_paths import ensure_model_slug_in_path
from rofa.core.runner import run_generation
from rofa.core.schemas import GenerationConfig
from rofa.papers.from_answers_to_hypotheses import config as default_paper_config
from rofa.papers.from_answers_to_hypotheses.methods import BranchSamplingEnsemble, GreedyDecode


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for generation."""
    parser = argparse.ArgumentParser(description="Generate ROFA run artifacts.")
    parser.add_argument("--model", required=True, help="Hugging Face model id.")
    parser.add_argument("--paper", default=default_paper_config.PAPER_ID)
    parser.add_argument("--method", help="greedy or branches (alias for k_sample_ensemble)")
    parser.add_argument("--out-dir", help="Output directory for the run artifacts.")
    parser.add_argument("--run-id", help="Optional run identifier appended to --out-dir.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume from existing progress.json when available.",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Expand an existing question set to the new --n without redoing prior examples.",
    )
    parser.add_argument("--question-set", dest="question_set_path")
    parser.add_argument("--create-question-set", action="store_true")
    parser.add_argument("--question-set-out", "--out", dest="question_set_out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--N", dest="n", type=int, help="Alias for --n.")
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-split")
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--k", dest="branches", type=int, help="Alias for --branches.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--hf-token", help="Optional Hugging Face access token.")
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable name for HF token (default: HF_TOKEN).",
    )
    return parser.parse_args()


def _maybe_create_question_set(args: argparse.Namespace) -> None:
    if not args.create_question_set:
        return
    qs_path = args.question_set_out or args.question_set_path
    if not qs_path:
        raise ValueError("--create-question-set requires --question-set-out or --question-set.")
    if os.path.exists(qs_path):
        qs = load_question_set(qs_path)
        print(f"Using existing question set at {qs_path} (qs_id={qs.qs_id})")
    else:
        qs = create_question_set(
            {"dataset_name": args.dataset_name, "dataset_split": args.dataset_split},
            {
                "seed": args.seed,
                "n": args.n,
                "subjects": args.subjects,
                "max_per_subject": args.n / args.subjects * 1.1 + 1,
            },
        )
        save_question_set(qs, qs_path)
        print(f"Saved question set to {qs_path} (qs_id={qs.qs_id})")
    if not args.question_set_path:
        args.question_set_path = qs_path


def _resolve_method_key(args: argparse.Namespace, paper) -> str:
    if not args.method:
        raise ValueError("--method is required when running generation.")
    method_key = resolve_method_key(args.method)
    if method_key not in paper.methods:
        aliases = list(list_method_aliases().keys())
        raise ValueError(
            f"--method must be one of {sorted(paper.methods.keys())} or aliases {aliases} "
            f"for paper {paper.paper_id}."
        )
    return method_key


def _build_method(method_key: str, args: argparse.Namespace):
    if method_key == "greedy":
        return GreedyDecode(), False
    return (
        BranchSamplingEnsemble(
            n_branches=args.branches,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        ),
        True,
    )


def _resolve_hf_token(args: argparse.Namespace) -> Optional[str]:
    if args.hf_token:
        return args.hf_token
    if args.hf_token_env:
        return os.getenv(args.hf_token_env)
    return None


def _maybe_login(model_id: str, hf_token: Optional[str]) -> None:
    if model_id.lower().startswith("google/medgemma") and not hf_token:
        raise ValueError(
            "This model may be gated on Hugging Face. Accept the model terms in the browser "
            "while logged in, then provide a HF token (read scope) via HF_TOKEN env var or "
            "--hf-token."
        )
    if hf_token:
        login(token=hf_token)


def _in_colab() -> bool:
    return bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"))


def _ensure_flash_attn() -> bool:
    if has_flash_attn():
        return True
    if not _in_colab():
        return False
    print("flash_attn not found; installing flash-attn (may require a runtime restart).")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "-q", "install", "flash-attn", "--no-build-isolation"],
            check=True,
        )
    except Exception as exc:  # noqa: BLE001
        print("flash_attn install failed; continuing without FlashAttention:", repr(exc))
        return False
    return has_flash_attn()


def _print_runtime_summary(model) -> None:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    dtype = next(model.parameters()).dtype
    gen_config = getattr(model, "generation_config", None)
    print("GPU:", gpu_name)
    print("dtype:", dtype)
    print("TORCHDYNAMO_DISABLE:", os.environ.get("TORCHDYNAMO_DISABLE"))
    print("use_cache:", getattr(model.config, "use_cache", None))
    print("cache_impl:", getattr(gen_config, "cache_implementation", None))
    print("attn_impl:", _resolve_attn_implementation(model))
    print("flash_attn_available:", has_flash_attn())


def _default_run_id(args: argparse.Namespace, method_key: str) -> str:
    k = args.branches if method_key != "greedy" else 1
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"N{args.n}_seed{args.seed}_k{k}_{timestamp}"


def _resolve_run_root(args: argparse.Namespace, model_slug: str, paper_id: str) -> str:
    base_dir = args.out_dir or os.path.join("runs", paper_id)
    return ensure_model_slug_in_path(base_dir, model_slug)


def main() -> None:
    """Run generation using a preselected paper/method configuration.

    Artifacts:
        Writes run artifacts under ``--out-dir`` (manifest, progress, question set,
        summary, and full records for k-sample ensemble).

    Failure modes:
        Raises ValueError for missing configuration, unsupported methods, or mismatched
        question sets when resuming.
    """
    args = parse_args()
    paper = get_paper(args.paper)
    args.dataset_name = args.dataset_name or paper.default_dataset
    args.dataset_split = args.dataset_split or paper.default_split
    _maybe_create_question_set(args)
    if args.create_question_set and not args.method:
        return
    method_key = _resolve_method_key(args, paper)
    model_id = args.model
    model_slug = to_slug(model_id)
    run_root = _resolve_run_root(args, model_slug, paper.paper_id)
    run_id = args.run_id or _default_run_id(args, method_key)
    run_dir = os.path.join(run_root, run_id)

    if args.resume:
        if not os.path.isdir(run_dir):
            raise ValueError(f"--resume set but run directory does not exist: {run_dir}")
    else:
        if os.path.exists(run_dir):
            raise ValueError(
                f"Run directory already exists: {run_dir}. Use --resume to continue."
            )

    hf_token = _resolve_hf_token(args)
    _maybe_login(model_id, hf_token)
    _ensure_flash_attn()

    try:
        tokenizer = load_tokenizer(model_id, hf_token=hf_token)
        model = load_model_with_fallback(model_id, hf_token=hf_token, tokenizer=tokenizer)
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if any(code in message for code in ["401", "403", "gated"]):
            raise RuntimeError(
                f"{message}\nIf you already provided a token, ensure you accepted the model’s "
                "access conditions on HF."
            ) from exc
        raise
    method, write_full_records = _build_method(method_key, args)
    _print_runtime_summary(model)

    config = GenerationConfig(
        method=method_key,
        model_id=model_id,
        model_slug=model_slug,
        out_dir=run_root,
        run_id=run_id,
        resume=args.resume,
        expand=args.expand,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        n=args.n,
        subjects=args.subjects,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        question_set_path=args.question_set_path,
        n_branches=args.branches if method_key != "greedy" else None,
        temperature=args.temperature if method_key != "greedy" else None,
        top_p=args.top_p if method_key != "greedy" else None,
        top_k=args.top_k if method_key != "greedy" else None,
        progress=True,
        write_full_records=write_full_records,
        tokenizer=tokenizer,
        model=model,
        method_impl=method,
    )
    run_generation(config)
    summary_path = os.path.join(run_dir, "summary.jsonl")
    manifest_path = os.path.join(run_dir, "manifest.json")
    question_set_path = os.path.join(run_dir, "question_set.json")
    artifacts = [summary_path, manifest_path, question_set_path]
    if write_full_records:
        artifacts.append(os.path.join(run_dir, "full.jsonl"))
    print(f"Run directory: {run_dir}")
    print("Artifacts:")
    for path in artifacts:
        print(f"  - {path}")
    return None


if __name__ == "__main__":
    main()
