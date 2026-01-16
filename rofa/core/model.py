"""Model and tokenizer loading for ROFA experiments."""

from __future__ import annotations

import inspect
import importlib.util
import os
import time
from typing import Callable, List, Optional, Tuple

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .parse import cop_to_letter, extract_choice_letter

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
    if attn_impl:
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = attn_impl
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = attn_impl
    return model


def has_flash_attn() -> bool:
    """Return True if flash_attn is importable."""
    return importlib.util.find_spec("flash_attn") is not None


def _resolve_attn_implementation(model) -> Optional[str]:
    """Return the configured attention implementation name when available."""
    config = getattr(model, "config", None)
    if config is None:
        return None
    for attr in ("attn_implementation", "_attn_implementation"):
        value = getattr(config, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


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
    if flash_available:
        try:
            model = load_model(model_id, "flash_attention_2", hf_token=hf_token)
            attn_impl = "flash_attention_2"
        except Exception as exc:  # noqa: BLE001
            print("FlashAttention2 failed, falling back to SDPA:", repr(exc))
            model = load_model(model_id, "sdpa", hf_token=hf_token)
            attn_impl = "sdpa"
    else:
        model = load_model(model_id, None, hf_token=hf_token)
    attn_label = attn_impl or _resolve_attn_implementation(model) or "default"
    print("Using attention implementation:", attn_label)
    print(
        "attn_implementation in config:",
        _resolve_attn_implementation(model),
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "use_cache"):
        model.generation_config.use_cache = True
    if _is_medgemma_model(model_id):
        if hasattr(model.config, "cache_implementation"):
            model.config.cache_implementation = "static"
        if hasattr(model, "generation_config"):
            model.generation_config.cache_implementation = "static"
    if tokenizer is not None:
        _configure_generation_padding(tokenizer, model)
    return model


def get_eos_ids(tokenizer) -> List[int]:
    """Return the list of EOS token ids, including <|eot_id|> when present."""
    eos_ids = [tokenizer.eos_token_id]
    try:
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot >= 0:
            eos_ids.append(eot)
    except Exception:  # noqa: BLE001
        pass
    return list({i for i in eos_ids if isinstance(i, int)})


def _apply_cache_implementation(model, cache_implementation: str) -> dict[str, object]:
    """Set supported cache implementation on config or generation kwargs."""
    kwargs: dict[str, object] = {}
    gen_config = getattr(model, "generation_config", None)
    if gen_config is not None and hasattr(gen_config, "cache_implementation"):
        gen_config.cache_implementation = cache_implementation
    signature = inspect.signature(model.generate)
    if "cache_implementation" in signature.parameters:
        kwargs["cache_implementation"] = cache_implementation
    return kwargs


def generate_greedy(
    model,
    inputs: dict,
    *,
    max_new_tokens: int,
    pad_token_id: int,
    eos_token_id: List[int] | int,
    cache_implementation: str = "static",
    log_assertions: bool = False,
):
    """Run deterministic greedy generation with safe defaults for MedGemma."""
    gen_kwargs: dict[str, object] = dict(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    gen_kwargs.update(_apply_cache_implementation(model, cache_implementation))
    if log_assertions:
        print(
            "Greedy generation kwargs:",
            {
                "do_sample": gen_kwargs.get("do_sample"),
                "num_beams": gen_kwargs.get("num_beams"),
                "max_new_tokens": gen_kwargs.get("max_new_tokens"),
                "cache_implementation": gen_kwargs.get("cache_implementation", None),
            },
        )
        assert gen_kwargs.get("do_sample") is False
        assert gen_kwargs.get("num_beams", 1) == 1
        assert "top_k" not in gen_kwargs and "top_p" not in gen_kwargs
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
        gen_kwargs: dict[str, object] = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_beams=1,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=eos_token_id,
        )
        model_id = _resolve_model_id(model)
        is_medgemma = _is_medgemma_model(model_id)
        if is_medgemma and hasattr(model, "generation_config"):
            model.generation_config.do_sample = True
            if hasattr(model.generation_config, "temperature"):
                model.generation_config.temperature = temperature
            if hasattr(model.generation_config, "top_p"):
                model.generation_config.top_p = top_p
            if hasattr(model.generation_config, "top_k"):
                model.generation_config.top_k = top_k
        else:
            gen_kwargs.update(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
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
