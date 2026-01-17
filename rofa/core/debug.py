"""Debug snapshot helpers for ROFA generation."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

import torch
import transformers
from transformers.utils import import_utils


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _print_kv(title: str, data: Dict[str, Any]) -> None:
    print(title)
    for key, value in data.items():
        print(f"  {key}: {value}")


def print_debug_snapshot(
    model,
    tokenizer,
    profile: Dict[str, Any],
    *,
    run_name: str,
    gen_kwargs: Dict[str, Any],
) -> None:
    """Print a debug snapshot once per run segment."""
    sys_info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    _print_kv(f"[debug:{run_name}] system", sys_info)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        gpu_info = {
            "name": torch.cuda.get_device_name(device),
            "capability": f"{props.major}.{props.minor}",
            "memory_total_gb": f"{props.total_memory / 1024**3:.2f}",
            "memory_allocated_mb": f"{torch.cuda.memory_allocated(device) / 1024**2:.2f}",
            "memory_reserved_mb": f"{torch.cuda.memory_reserved(device) / 1024**2:.2f}",
            "memory_max_reserved_mb": f"{torch.cuda.max_memory_reserved(device) / 1024**2:.2f}",
        }
    else:
        gpu_info = {"name": "cpu"}
    _print_kv(f"[debug:{run_name}] gpu", gpu_info)

    feature_flags = {
        "flash_attn_2_available": import_utils.is_flash_attn_2_available(),
        "torch_sdpa_available": import_utils.is_torch_sdpa_available(),
    }
    _print_kv(f"[debug:{run_name}] feature_flags", feature_flags)

    param_count = sum(p.numel() for p in model.parameters())
    model_info = {
        "class": model.__class__.__name__,
        "dtype": str(next(model.parameters()).dtype),
        "param_count_m": f"{param_count / 1e6:.2f}",
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "profile_dtype": profile.get("dtype"),
        "profile_attn_implementation": profile.get("attn_implementation"),
        "profile_notes": profile.get("notes"),
    }
    _print_kv(f"[debug:{run_name}] model", model_info)

    gen_config = getattr(model, "generation_config", None)
    gen_info = {
        "do_sample": getattr(gen_config, "do_sample", None),
        "top_k": getattr(gen_config, "top_k", None),
        "top_p": getattr(gen_config, "top_p", None),
        "temperature": getattr(gen_config, "temperature", None),
        "cache_implementation": getattr(gen_config, "cache_implementation", None),
    }
    _print_kv(f"[debug:{run_name}] generation_config", gen_info)

    safe_kwargs = _json_safe(gen_kwargs)
    print(f"[debug:{run_name}] run_kwargs: {json.dumps(safe_kwargs, indent=2)}")

    try:
        sample = tokenizer("cache probe", return_tensors="pt")
        sample = {k: v.to(model.device) for k, v in sample.items()}
        with torch.inference_mode():
            outputs = model(**sample, use_cache=True)
        print(
            f"[debug:{run_name}] cache_probe_past_key_values_type: "
            f"{type(outputs.past_key_values)}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[debug:{run_name}] cache_probe_error: {exc!r}")
