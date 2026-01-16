"""Model and tokenizer loading for ROFA experiments."""

from __future__ import annotations

import copy
import time
from typing import Callable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .parse import cop_to_letter, extract_choice_letter

DEFAULT_MODEL_ID = "HPAI-BSC/Llama3.1-Aloe-Beta-8B"


def load_tokenizer(model_id: str, hf_token: Optional[str] = None):
    """Load the tokenizer for the model."""
    return AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)


def load_model(model_id: str, attn_impl: str, hf_token: Optional[str] = None):
    """Load the model with the requested attention implementation."""
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=attn_impl,  # "flash_attention_2" или "sdpa"
        token=hf_token,
    ).eval()


def load_model_with_fallback(model_id: str, hf_token: Optional[str] = None):
    """Load the model, preferring FlashAttention2 and falling back to SDPA."""
    try:
        model = load_model(model_id, "flash_attention_2", hf_token=hf_token)
        attn_impl = "flash_attention_2"
    except Exception as exc:  # noqa: BLE001
        print("FlashAttention2 failed, falling back to SDPA:", repr(exc))
        model = load_model(model_id, "sdpa", hf_token=hf_token)
        attn_impl = "sdpa"
    print("Using", "FlashAttention2" if attn_impl == "flash_attention_2" else "SDPA")
    print("attn_implementation in config:", getattr(model.config, "_attn_implementation", None))
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

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

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
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    gen_kwargs: dict[str, object] = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=1,
        use_cache=True,
    )

    if do_sample:
        gen_kwargs.update(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    else:
        gen_config = copy.deepcopy(getattr(model, "generation_config", None))
        if gen_config is not None:
            gen_config.do_sample = False
            gen_config.temperature = 1.0
            gen_config.top_p = 1.0
            gen_config.top_k = 50
            gen_kwargs["generation_config"] = gen_config

    with torch.no_grad():
        out = model.generate(**gen_kwargs)

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
