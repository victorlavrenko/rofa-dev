"""Run CGVF-lite controller over k-sample branch outputs."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterator, Optional

import torch

from rofa.core.accel import load_model_and_tokenizer
from rofa.core.cgvf_lite import (
    CgvfLiteConfig,
    VerifyResult,
    parse_verify_response,
    run_cgvf_lite,
)
from rofa.core.generation import build_prompt
from rofa.core.tokens import get_eos_ids

SYSTEM_PROMPT = "You are a careful medical exam checker. Your job is to verify a proposed candidate option."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CGVF-lite controller with verify-first calls.")
    parser.add_argument("--input", required=True, help="Path to full.jsonl with branches.")
    parser.add_argument("--output", required=True, help="Path to write CGVF-lite logs (JSONL).")
    parser.add_argument("--model-id", help="HF model id for verification calls.")
    parser.add_argument("--hf-token", help="HF token (or set HUGGINGFACE_TOKEN).")
    parser.add_argument("--attn-impl", choices=["flash_attention_2", "sdpa", "eager"], help="Attention impl override.")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], help="Torch dtype override.")
    parser.add_argument("--limit", type=int, help="Optional max number of questions to process.")
    return parser.parse_args()


def _resolve_hf_token(args: argparse.Namespace) -> Optional[str]:
    if args.hf_token:
        return args.hf_token
    return os.environ.get("HUGGINGFACE_TOKEN")


def _load_records(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc


def _build_verify_prompt(
    tokenizer,
    question: str,
    options: Dict[str, str],
    rewritten_summary: str,
    candidate: str,
) -> str:
    user_prompt = (
        f"Question:\n{question}\n\n"
        "Options:\n"
        f"A) {options.get('A', '')}\n"
        f"B) {options.get('B', '')}\n"
        f"C) {options.get('C', '')}\n"
        f"D) {options.get('D', '')}\n\n"
        "Evidence from multiple independent attempts (may disagree):\n"
        f"{rewritten_summary}\n\n"
        f"Candidate to verify: {candidate}\n\n"
        "Task:\n"
        f"1) Decide if Candidate {candidate} is correct.\n"
        "2) If incorrect, output the single best correct option among A/B/C/D.\n"
        "3) Output ONLY in this exact JSON, on one line:\n"
        '{"is_correct": true/false, "final": "A|B|C|D", "reason": "<max 2 sentences>"}'
    )
    return build_prompt(tokenizer, SYSTEM_PROMPT, user_prompt)


def _seed_all(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _verify_with_model(
    *,
    question: str,
    options: Dict[str, str],
    rewritten_summary: str,
    candidate: str,
    seed: int,
    tokenizer,
    model,
) -> Optional[VerifyResult]:
    _seed_all(seed)
    prompt = _build_verify_prompt(tokenizer, question, options, rewritten_summary, candidate)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    eos_ids = get_eos_ids(tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_ids[0] if eos_ids else tokenizer.eos_token_id
    if pad_id is not None:
        model.config.pad_token_id = pad_id
        model.generation_config.pad_token_id = pad_id
    gen_kwargs = {
        **inputs,
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 1.0,
        "top_k": 0,
        "max_new_tokens": 140,
        "pad_token_id": pad_id,
        "eos_token_id": eos_ids or tokenizer.eos_token_id,
    }
    with torch.inference_mode():
        out = model.generate(**gen_kwargs)
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return parse_verify_response(decoded)


if __name__ == "__main__":
    args = parse_args()
    hf_token = _resolve_hf_token(args)
    model_id = args.model_id

    if model_id is None:
        raise SystemExit("--model-id is required to run verification calls.")

    overrides = {"attn_implementation": args.attn_impl, "dtype": args.dtype}
    tokenizer, model, _profile = load_model_and_tokenizer(
        model_id,
        hf_token=hf_token,
        overrides=overrides,
    )

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    processed = 0
    with open(output_path, "w", encoding="utf-8") as out_handle:
        for record in _load_records(args.input):
            question = record.get("question")
            options = record.get("options")
            branches = record.get("branches")
            if not question or not isinstance(options, dict) or not isinstance(branches, list):
                raise ValueError("Record missing question/options/branches in full.jsonl input.")

            def verifier(candidate: str, seed: int, rewritten_summary: str):
                return _verify_with_model(
                    question=question,
                    options=options,
                    rewritten_summary=rewritten_summary,
                    candidate=candidate,
                    seed=seed,
                    tokenizer=tokenizer,
                    model=model,
                )

            final_pred, log_record = run_cgvf_lite(
                question=question,
                options=options,
                branches=branches,
                verify_fn=verifier,
                config=CgvfLiteConfig(),
            )
            log_record.update(
                {
                    "index": record.get("index"),
                    "id": record.get("id"),
                }
            )
            out_handle.write(json.dumps(log_record) + "\n")
            processed += 1
            if args.limit and processed >= args.limit:
                break

    print(f"Wrote {processed} CGVF-lite records to {output_path}")
