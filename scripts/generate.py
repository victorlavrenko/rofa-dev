"""CLI entrypoint for generation runs."""

from __future__ import annotations

import argparse
import os
import uuid

from rofa.io import _now_utc, load_progress, write_manifest
from rofa.methods import BranchSamplingEnsemble, GreedyDecode
from rofa.model import MODEL_ID, load_model_with_fallback, load_tokenizer
from rofa.runner import run_dataset_loop
from rofa.schemas import RunConfig, RunManifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ROFA run artifacts.")
    parser.add_argument("--method", choices=["greedy", "branches"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume from existing progress.json when available.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--dataset-name", default="openlifescienceai/medmcqa")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or uuid.uuid4().hex
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    progress_path = os.path.join(run_dir, "progress.json")
    manifest_path = os.path.join(run_dir, "manifest.json")
    summary_path = os.path.join(run_dir, "summary.jsonl")
    full_path = os.path.join(run_dir, "full.jsonl")

    progress = load_progress(progress_path)
    resume = args.resume if args.resume is not None else bool(progress)
    if progress:
        progress_run_id = progress.get("run_id")
        if progress_run_id and progress_run_id != run_id:
            raise ValueError(
                f"Run ID mismatch: progress has {progress_run_id} but --run-id is {run_id}."
            )
        if not resume:
            raise ValueError(
                "progress.json exists but --no-resume was set; refuse to overwrite artifacts."
            )
    else:
        if resume and os.path.exists(summary_path):
            raise ValueError(
                "summary.jsonl exists but no progress.json was found; cannot safely resume."
            )

    tokenizer = load_tokenizer()
    model = load_model_with_fallback()

    if args.method == "greedy":
        method = GreedyDecode()
        full_output = None
    else:
        method = BranchSamplingEnsemble(
            n_branches=args.branches,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        full_output = full_path

    config = RunConfig(
        method=args.method,
        model_id=MODEL_ID,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        n=args.n,
        subjects=args.subjects,
        max_per_subject=args.n / args.subjects * 1.1 + 1,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        n_branches=args.branches if args.method == "branches" else None,
        temperature=args.temperature if args.method == "branches" else None,
        top_p=args.top_p if args.method == "branches" else None,
        top_k=args.top_k if args.method == "branches" else None,
    )

    if not os.path.exists(manifest_path):
        manifest = RunManifest(
            run_id=run_id,
            created_at=_now_utc(),
            method=args.method,
            config=config,
        )
        write_manifest(manifest_path, manifest)

    run_dataset_loop(
        method=method,
        output_summary_path=summary_path,
        output_full_path=full_output,
        progress_path=progress_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        seed=args.seed,
        n=args.n,
        subjects=args.subjects,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
        model=model,
        run_id=run_id,
    )


if __name__ == "__main__":
    main()
