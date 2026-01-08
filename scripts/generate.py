"""CLI entrypoint for generation runs."""

from __future__ import annotations

import argparse
import os

from rofa.methods import BranchSamplingEnsemble, GreedyDecode
from rofa.model import MODEL_ID, load_model_with_fallback, load_tokenizer
from rofa.question_set import create_question_set, load_question_set, save_question_set
from rofa.runner import run_generation
from rofa.schemas import GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ROFA run artifacts.")
    parser.add_argument("--method", choices=["greedy", "branches"])
    parser.add_argument("--out-dir")
    parser.add_argument("--run-id")
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
    parser.add_argument("--dataset-name", default="openlifescienceai/medmcqa")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.create_question_set:
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
        if not args.method:
            return
        if not args.question_set_path:
            args.question_set_path = qs_path

    if not args.method:
        raise ValueError("--method is required when running generation.")
    if not args.out_dir:
        raise ValueError("--out-dir is required when running generation.")

    tokenizer = load_tokenizer()
    model = load_model_with_fallback()

    if args.method == "greedy":
        method = GreedyDecode()
        write_full_records = False
    else:
        method = BranchSamplingEnsemble(
            n_branches=args.branches,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        write_full_records = True

    config = GenerationConfig(
        method=args.method,
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
        n_branches=args.branches if args.method == "branches" else None,
        temperature=args.temperature if args.method == "branches" else None,
        top_p=args.top_p if args.method == "branches" else None,
        top_k=args.top_k if args.method == "branches" else None,
        progress=True,
        heartbeat_every=10,
        write_full_records=write_full_records,
        tokenizer=tokenizer,
        model=model,
        method_impl=method,
    )

    run_generation(config)


if __name__ == "__main__":
    main()
