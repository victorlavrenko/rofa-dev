"""CLI entrypoint for generation runs."""

from __future__ import annotations

import argparse
import os

from rofa.core.model import MODEL_ID, load_model_with_fallback, load_tokenizer
from rofa.core.question_set import create_question_set, load_question_set, save_question_set
from rofa.core.registry import get_paper, list_method_aliases, resolve_method_key
from rofa.core.runner import run_generation
from rofa.core.schemas import GenerationConfig
from rofa.papers.from_answers_to_hypotheses import config as default_paper_config
from rofa.papers.from_answers_to_hypotheses.methods import BranchSamplingEnsemble, GreedyDecode


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for generation."""
    parser = argparse.ArgumentParser(description="Generate ROFA run artifacts.")
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
    parser.add_argument("--question-set", dest="question_set_path")
    parser.add_argument("--create-question-set", action="store_true")
    parser.add_argument("--question-set-out", "--out", dest="question_set_out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-split")
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
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
    if not args.out_dir:
        raise ValueError("--out-dir is required when running generation.")

    tokenizer = load_tokenizer()
    model = load_model_with_fallback()
    method, write_full_records = _build_method(method_key, args)

    config = GenerationConfig(
        method=method_key,
        model_id=MODEL_ID,
        out_dir=args.out_dir,
        run_id=args.run_id,
        resume=args.resume,
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
        heartbeat_every=10,
        write_full_records=write_full_records,
        tokenizer=tokenizer,
        model=model,
        method_impl=method,
    )
    run_generation(config)
    run_dir = os.path.join(args.out_dir, args.run_id) if args.run_id else args.out_dir
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
