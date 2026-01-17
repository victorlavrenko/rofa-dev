"""Batched generation utilities for ROFA runs."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from transformers.utils import import_utils

from rofa.core.debug import print_debug_snapshot
from rofa.core.tokens import get_eos_ids


def build_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001
            pass
    return f"{system_prompt}\n\n{user_prompt}"


def build_greedy_kwargs(max_new_tokens: int) -> Dict[str, Any]:
    return {
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "max_new_tokens": max_new_tokens,
    }


def assert_greedy_kwargs(kwargs: Dict[str, Any]) -> None:
    assert kwargs.get("do_sample") is False
    assert kwargs.get("num_beams", 1) == 1
    assert "top_k" not in kwargs
    assert "top_p" not in kwargs
    assert "temperature" not in kwargs


def build_sampling_kwargs(
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Dict[str, Any]:
    return {
        "do_sample": True,
        "num_beams": 1,
        "use_cache": True,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }


def sdpa_fast_ctx():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        )
    except Exception:  # noqa: BLE001
        return nullcontext()


def _tokenize_prompts(tokenizer, prompts: List[str], device: torch.device):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    input_lens = enc["attention_mask"].sum(dim=1).tolist()
    return enc, input_lens


def _decode_outputs(tokenizer, out_ids: torch.Tensor, input_lens: List[int]) -> List[str]:
    decoded: List[str] = []
    for i, input_len in enumerate(input_lens):
        gen_part = out_ids[i, input_len:]
        decoded.append(tokenizer.decode(gen_part, skip_special_tokens=True))
    return decoded


def _is_cuda_oom(exc: Exception) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    return "CUDA out of memory" in str(exc) or "cuda out of memory" in str(exc).lower()


def _ensure_pad_token(tokenizer, model) -> Tuple[int, List[int] | int]:
    eos_ids = get_eos_ids(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_ids[0] if eos_ids else tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer.pad_token_id, eos_ids or tokenizer.eos_token_id


def _build_sdpa_context(profile: Dict[str, Any]):
    if profile.get("attn_implementation") == "sdpa" and import_utils.is_torch_sdpa_available():
        return sdpa_fast_ctx()
    return nullcontext()


def _normalize_items(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for item in items:
        if "item_id" not in item or "prompt" not in item:
            raise ValueError("Each item must include item_id and prompt.")
        normalized.append(item)
    return normalized


def run_greedy_batched(
    items: Iterable[Dict[str, Any]],
    *,
    tokenizer,
    model,
    batch_size_q: int,
    greedy_kwargs: Dict[str, Any],
    profile: Dict[str, Any],
    on_chunk: Optional[Callable[[List[Dict[str, Any]], List[str], float], None]] = None,
) -> Tuple[Dict[Any, str], float]:
    items = _normalize_items(items)
    if batch_size_q < 1:
        raise ValueError("batch_size_q must be >= 1")
    print_debug_snapshot(model, tokenizer, profile, run_name="greedy", gen_kwargs=greedy_kwargs)
    assert_greedy_kwargs(greedy_kwargs)
    pad_token_id, eos_token_id = _ensure_pad_token(tokenizer, model)
    device = model.device
    sdpa_ctx = _build_sdpa_context(profile)
    outputs: Dict[Any, str] = {}
    idx = 0
    current_batch = batch_size_q
    warned = False
    total_time = 0.0

    while idx < len(items):
        chunk = items[idx : idx + current_batch]
        prompts = [item["prompt"] for item in chunk]
        enc, input_lens = _tokenize_prompts(tokenizer, prompts, device)
        gen_kwargs = {
            **enc,
            **greedy_kwargs,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        }
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode(), sdpa_ctx:
                out_ids = model.generate(**gen_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            chunk_time = time.time() - t0
            total_time += chunk_time
        except Exception as exc:  # noqa: BLE001
            if _is_cuda_oom(exc) and current_batch > 1:
                current_batch = max(1, current_batch // 2)
                if not warned:
                    print(f"Warning: CUDA OOM in greedy; reducing batch_size_q to {current_batch}.")
                    warned = True
                torch.cuda.empty_cache()
                continue
            raise
        decoded = _decode_outputs(tokenizer, out_ids, input_lens)
        if on_chunk is not None:
            on_chunk(chunk, decoded, chunk_time)
        for item, text in zip(chunk, decoded):
            outputs[item["item_id"]] = text
        idx += len(chunk)

    return outputs, total_time


def _build_generators(seed: Optional[int], count: int, device: torch.device, offset: int):
    if seed is None:
        return None
    generators = []
    for i in range(count):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + offset + i)
        generators.append(gen)
    return generators


def run_ensemble_batched(
    items: Iterable[Dict[str, Any]],
    *,
    tokenizer,
    model,
    batch_size_q: int,
    n_branches: int,
    branch_batch_size: int,
    sample_kwargs: Dict[str, Any],
    profile: Dict[str, Any],
    seed: Optional[int] = None,
    on_chunk: Optional[
        Callable[
            [List[Dict[str, Any]], Dict[Any, List[str]], Dict[Any, List[Optional[int]]], float],
            None,
        ]
    ] = None,
) -> Tuple[Dict[Any, List[str]], Dict[Any, List[Optional[int]]], float]:
    items = _normalize_items(items)
    if batch_size_q < 1:
        raise ValueError("batch_size_q must be >= 1")
    if not (1 <= branch_batch_size <= n_branches):
        raise ValueError("branch_batch_size must satisfy 1 <= branch_batch_size <= n_branches")
    print_debug_snapshot(model, tokenizer, profile, run_name="ensemble", gen_kwargs=sample_kwargs)
    pad_token_id, eos_token_id = _ensure_pad_token(tokenizer, model)
    device = model.device
    sdpa_ctx = _build_sdpa_context(profile)
    outputs: Dict[Any, List[str]] = {
        item["item_id"]: ["" for _ in range(n_branches)] for item in items
    }
    branch_seeds: Dict[Any, List[Optional[int]]] = {
        item["item_id"]: [None for _ in range(n_branches)] for item in items
    }
    idx = 0
    current_batch = batch_size_q
    current_branch_batch = branch_batch_size
    warned_q = False
    warned_branch = False
    total_time = 0.0
    global_row_offset = 0

    while idx < len(items):
        chunk = items[idx : idx + current_batch]
        branch_offset = 0
        chunk_time = 0.0

        while branch_offset < n_branches:
            batch_branch = min(current_branch_batch, n_branches - branch_offset)
            prompts = []
            mapping: List[Tuple[Any, int]] = []
            for item in chunk:
                for b in range(branch_offset, branch_offset + batch_branch):
                    prompts.append(item["prompt"])
                    mapping.append((item["item_id"], b))
            enc, input_lens = _tokenize_prompts(tokenizer, prompts, device)
            gen_kwargs = {
                **enc,
                **sample_kwargs,
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
            }
            generators = _build_generators(seed, len(prompts), device, global_row_offset)
            used_row_generators = generators is not None
            try:
                if generators is not None:
                    gen_kwargs["generator"] = generators
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
                with torch.inference_mode(), sdpa_ctx:
                    out_ids = model.generate(**gen_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_time = time.time() - t0
                total_time += batch_time
                chunk_time += batch_time
            except Exception as exc:  # noqa: BLE001
                if generators is not None and isinstance(exc, (TypeError, ValueError)):
                    gen_kwargs.pop("generator", None)
                    generators = None
                    used_row_generators = False
                    if seed is not None:
                        torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.time()
                    with torch.inference_mode(), sdpa_ctx:
                        out_ids = model.generate(**gen_kwargs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    batch_time = time.time() - t0
                    total_time += batch_time
                    chunk_time += batch_time
                elif _is_cuda_oom(exc):
                    if current_batch > 1:
                        current_batch = max(1, current_batch // 2)
                        if not warned_q:
                            print(
                                "Warning: CUDA OOM in ensemble; reducing "
                                f"batch_size_q to {current_batch}."
                            )
                            warned_q = True
                        torch.cuda.empty_cache()
                        break
                    if current_branch_batch > 1:
                        current_branch_batch = max(1, current_branch_batch // 2)
                        if not warned_branch:
                            print(
                                "Warning: CUDA OOM in ensemble; reducing "
                                f"branch_batch_size to {current_branch_batch}."
                            )
                            warned_branch = True
                        torch.cuda.empty_cache()
                        break
                    raise
                else:
                    raise

            decoded = _decode_outputs(tokenizer, out_ids, input_lens)
            for row_idx, ((item_id, branch_idx), text) in enumerate(zip(mapping, decoded)):
                outputs[item_id][branch_idx] = text
                if seed is not None:
                    if used_row_generators:
                        branch_seeds[item_id][branch_idx] = seed + global_row_offset + row_idx
                    else:
                        branch_seeds[item_id][branch_idx] = seed
            global_row_offset += len(prompts)
            branch_offset += batch_branch
        if branch_offset >= n_branches:
            if on_chunk is not None:
                chunk_outputs = {item["item_id"]: outputs[item["item_id"]] for item in chunk}
                chunk_seeds = {item["item_id"]: branch_seeds[item["item_id"]] for item in chunk}
                on_chunk(chunk, chunk_outputs, chunk_seeds, chunk_time)
            idx += len(chunk)

    return outputs, branch_seeds, total_time
