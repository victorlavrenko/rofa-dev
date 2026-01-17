"""Model acceleration utilities for A100-generation runs."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import import_utils

KNOWN_MODEL_IDS = {
    "HPAI-BSC/Llama3.1-Aloe-Beta-8B",
    "HPAI-BSC/Qwen2.5-Aloe-Beta-7B",
    "m42-health/Llama3-Med42-8B",
    "BioMistral/BioMistral-7B",
    "google/medgemma-1.5-4b-it",
}


def _resolve_dtype() -> str:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float16"


def _choose_attn_impl(*, prefer_flash: bool) -> Optional[str]:
    flash_ok = import_utils.is_flash_attn_2_available()
    sdpa_ok = import_utils.is_torch_sdpa_available()
    if prefer_flash and flash_ok:
        return "flash_attention_2"
    if sdpa_ok:
        return "sdpa"
    return "eager"


def select_model_profile(model_id: str) -> Dict[str, Any]:
    """Select attention and dtype settings for a given model id."""
    dtype = _resolve_dtype()
    if model_id == "google/medgemma-1.5-4b-it":
        return {
            "dtype": dtype,
            "attn_implementation": "sdpa",
            "supports_branch_batching": True,
            "notes": "MedGemma prefers SDPA on A100; static cache not forced.",
        }
    if model_id in {
        "HPAI-BSC/Llama3.1-Aloe-Beta-8B",
        "m42-health/Llama3-Med42-8B",
    }:
        return {
            "dtype": dtype,
            "attn_implementation": _choose_attn_impl(prefer_flash=True),
            "supports_branch_batching": True,
            "notes": "Llama family: FA2 when available, otherwise SDPA.",
        }
    if model_id == "HPAI-BSC/Qwen2.5-Aloe-Beta-7B":
        return {
            "dtype": dtype,
            "attn_implementation": _choose_attn_impl(prefer_flash=True),
            "supports_branch_batching": True,
            "notes": "Qwen2.5: FA2 when available, otherwise SDPA.",
        }
    if model_id == "BioMistral/BioMistral-7B":
        return {
            "dtype": dtype,
            "attn_implementation": _choose_attn_impl(prefer_flash=True),
            "supports_branch_batching": True,
            "notes": "Mistral family: FA2 when available, otherwise SDPA.",
        }
    return {
        "dtype": dtype,
        "attn_implementation": _choose_attn_impl(prefer_flash=True),
        "supports_branch_batching": True,
        "notes": "Generic A100 profile.",
    }


def apply_a100_runtime_settings(profile: Dict[str, Any]) -> None:
    """Apply runtime settings for A100 generation."""
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    torch.set_float32_matmul_precision("high")
    _ = profile


def load_model_and_tokenizer(
    model_id: str,
    hf_token: Optional[str] = None,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load tokenizer/model with A100-aware defaults."""
    profile = select_model_profile(model_id)
    if overrides:
        profile = {**profile, **{k: v for k, v in overrides.items() if v is not None}}
    apply_a100_runtime_settings(profile)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)
    dtype = torch.bfloat16 if profile["dtype"] == "bfloat16" else torch.float16
    kwargs = {
        "torch_dtype": dtype,
        "device_map": "cuda",
        "low_cpu_mem_usage": True,
        "token": hf_token,
    }
    attn_impl = profile.get("attn_implementation")
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model, profile


def resolve_default_batch_sizes(model_id: str, n_branches: int) -> Dict[str, int]:
    """Return default batch sizing per model id."""
    defaults = {
        "google/medgemma-1.5-4b-it": (8, 2, 5),
        "HPAI-BSC/Llama3.1-Aloe-Beta-8B": (4, 1, 5),
        "m42-health/Llama3-Med42-8B": (4, 1, 5),
        "HPAI-BSC/Qwen2.5-Aloe-Beta-7B": (4, 1, 5),
        "BioMistral/BioMistral-7B": (4, 1, 5),
    }
    greedy, ensemble_q, branch = defaults.get(model_id, (4, 1, 5))
    branch = max(1, min(branch, n_branches))
    return {
        "greedy_batch_size": greedy,
        "ensemble_batch_size_q": ensemble_q,
        "branch_batch_size": branch,
    }
