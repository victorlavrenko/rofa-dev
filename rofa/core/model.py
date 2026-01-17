"""Model and tokenizer loading for ROFA experiments."""

from __future__ import annotations

import importlib.util
import os
import time
from typing import Callable, List, Optional, Tuple

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .generation import assert_greedy_kwargs, build_greedy_kwargs, build_sampling_kwargs
from .parse import cop_to_letter, extract_choice_letter
from .tokens import get_eos_ids

DEFAULT_MODEL_ID = "HPAI-BSC/Llama3.1-Aloe-Beta-8B"

torch.set_float32_matmul_precision("high")


def load_tokenizer(model_id: str, hf_token: Optional[str] = None):
    """Load the tokenizer for the model."""
    return AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)


def load_model(model_id: str, attn_impl: Optional[str], hf_token: Optional[str] = None):
    """Load the model with the requested attention implementation."""
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda",
        "low_cpu_mem_usage": True,
        "token": hf_token,
    }
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl  # "flash_attention_2" или "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    if attn_impl and hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = attn_impl
    return model


def has_flash_attn() -> bool:
    """Return True if flash_attn is importable."""
    return importlib.util.find_spec("flash_attn") is not None


def _is_medgemma_model(model_id: Optional[str]) -> bool:
    """Return True when the model id refers to MedGemma variants."""
    if not model_id:
        return False
    return model_id.lower().startswith("google/medgemma")


def _resolve_model_id(model) -> str:
    """Resolve the model id/name from common model attributes."""
    for attr in ("name_or_path",):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    config = getattr(model, "config", None)
    for attr in ("name_or_path", "_name_or_path", "model_id"):
        value = getattr(config, attr, None) if config is not None else None
        if isinstance(value, str) and value:
            return value
    return ""


def _configure_generation_padding(tokenizer, model) -> None:
    """Ensure pad/eos ids are consistent to avoid repeated Transformers warnings."""
    eos_ids = get_eos_ids(tokenizer)
    if tokenizer.pad_token_id is None:
        if eos_ids:
            tokenizer.pad_token_id = eos_ids[0]
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id


def load_model_with_fallback(
    model_id: str, hf_token: Optional[str] = None, tokenizer=None
):
    """Load the model, preferring FlashAttention2 and falling back to SDPA."""
    flash_available = has_flash_attn()
    attn_impl: Optional[str] = None
    if _is_medgemma_model(model_id):
        model = load_model(model_id, "sdpa", hf_token=hf_token)
        attn_impl = "sdpa"
    elif flash_available:
        try:
            model = load_model(model_id, "flash_attention_2", hf_token=hf_token)
            attn_impl = "flash_attention_2"
        except Exception as exc:  # noqa: BLE001
            print("FlashAttention2 failed, falling back to SDPA:", repr(exc))
            model = load_model(model_id, "sdpa", hf_token=hf_token)
            attn_impl = "sdpa"
    else:
        model = load_model(model_id, None, hf_token=hf_token)
    attn_label = attn_impl or getattr(model.config, "attn_implementation", None) or "default"
    print("Using attention implementation:", attn_label)
    print(
        "attn_implementation in config:",
        getattr(model.config, "attn_implementation", None),
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "use_cache"):
        model.generation_config.use_cache = True
    if tokenizer is not None:
        _configure_generation_padding(tokenizer, model)
    return model



def generate_greedy(
    model,
    inputs: dict,
    *,
    max_new_tokens: int,
    pad_token_id: int,
    eos_token_id: List[int] | int,
    log_assertions: bool = False,
):
    """Run deterministic greedy generation with safe defaults for MedGemma."""
    gen_kwargs: dict[str, object] = {
        **inputs,
        **build_greedy_kwargs(max_new_tokens=max_new_tokens),
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    if log_assertions:
        print("Greedy generation kwargs:", {k: gen_kwargs.get(k) for k in gen_kwargs})
        assert_greedy_kwargs(gen_kwargs)
    return model.generate(**gen_kwargs)


def infer_one(
    example,
    tokenizer,
    model,
    max_new_tokens: int = 512,
    *,
    system_prompt: str,
    build_user_prompt: Callable[[dict], str],
    seed: Optional[int] = None,
    temperature: Optional[float] = None,
    do_sample: Optional[bool] = None,
    top_p: float = 1.0,
    top_k: int = 0,
    log_greedy: bool = False,
) -> Tuple[Optional[str], str, str, float]:
    """
    Backward-compatible inference.

    - If called with do_sample=None, behaves like the old version: greedy.
    - If do_sample=True, uses sampling with the provided temperature and seed.
    """
    if seed is not None:
        import os
        import random

        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if do_sample is None:
        do_sample = False

    if temperature is None:
        temperature = 1.0

    _configure_generation_padding(tokenizer, model)

    user_prompt = build_user_prompt(example)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = None
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001
            prompt = None
    if prompt is None:
        prompt = f"{system_prompt}\n\n{user_prompt}"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    eos_ids = get_eos_ids(tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_ids[0] if eos_ids else tokenizer.eos_token_id
    eos_token_id = eos_ids or tokenizer.eos_token_id
    # Pass pad/eos ids explicitly to avoid repeated Transformers fallback warnings.
    if do_sample:
        gen_kwargs: dict[str, object] = {
            **inputs,
            **build_sampling_kwargs(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ),
            "pad_token_id": pad_id,
            "eos_token_id": eos_token_id,
        }
        with torch.inference_mode():
            out = model.generate(**gen_kwargs)
    else:
        with torch.inference_mode():
            out = generate_greedy(
                model,
                inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                eos_token_id=eos_token_id,
                log_assertions=log_greedy,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    options = {
        "A": example.get("opa"),
        "B": example.get("opb"),
        "C": example.get("opc"),
        "D": example.get("opd"),
    }
    pred = extract_choice_letter(gen, options=options)
    gold = cop_to_letter(example["cop"])
    return pred, gold, gen, t1 - t0
