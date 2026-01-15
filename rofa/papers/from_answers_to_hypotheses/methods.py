"""Inference methods for ROFA experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from rofa.core.io import _now_utc
from rofa.core.metrics import _correct_fraction, _diversity_metrics
from rofa.core.model import infer_one
from rofa.papers.from_answers_to_hypotheses.prompts import SYSTEM, build_user


class MethodProtocol(Protocol):
    """Protocol for paper method implementations."""

    def run_one(self, example: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the method on a single example and return a summary record."""
        ...


@dataclass
class GreedyDecode:
    """Single greedy generation with direct parsing."""

    def run_one(self, example: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        tokenizer = context["tokenizer"]
        model = context["model"]
        max_new_tokens = context["max_new_tokens"]
        seed = context["seed"]
        index = context["index"]
        subject_name = context["subject_name"]
        model_id = context["model_id"]
        model_slug = context["model_slug"]

        pred, gold, gen, inference_time = infer_one(
            example,
            tokenizer,
            model,
            system_prompt=SYSTEM,
            build_user_prompt=build_user,
            max_new_tokens=max_new_tokens,
        )

        return {
            "index": index,
            "id": example.get("id"),
            "question": example.get("question"),
            "options": {
                "A": example.get("opa"),
                "B": example.get("opb"),
                "C": example.get("opc"),
                "D": example.get("opd"),
            },
            "gold": gold,
            "prediction": pred,
            "is_correct": (pred == gold) if pred is not None else None,
            "model_output": gen,
            "subject_name": subject_name,
            "inference_time_sec": inference_time,
            "model_id": model_id,
            "model_slug": model_slug,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "timestamp": _now_utc(),
        }


@dataclass
class BranchSamplingEnsemble:
    """Multi-sample (k-sample) ensemble with per-branch consensus metrics."""

    n_branches: int = 10
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 50

    last_full_record: Optional[Dict[str, Any]] = None

    def run_one(self, example: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        tokenizer = context["tokenizer"]
        model = context["model"]
        max_new_tokens = context["max_new_tokens"]
        seed = context["seed"]
        index = context["index"]
        picked_index = context["picked_index"]
        subject_name = context["subject_name"]
        model_id = context["model_id"]
        model_slug = context["model_slug"]

        gold = context["gold"]

        preds: List[Optional[str]] = []
        branch_times: List[float] = []
        branch_records: List[Dict[str, Any]] = []

        t0 = time.time()
        example_seed_offset = 100_000 * picked_index

        for j in range(self.n_branches):
            branch_seed = seed + example_seed_offset + 10_000 + j * 997

            pred, _, gen, dt = infer_one(
                example,
                tokenizer,
                model,
                system_prompt=SYSTEM,
                build_user_prompt=build_user,
                seed=branch_seed,
                temperature=self.temperature,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=self.top_p,
                top_k=self.top_k,
            )

            preds.append(pred)
            branch_times.append(dt)

            branch_records.append(
                {
                    "branch": j,
                    "seed": branch_seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "pred": pred,
                    "gold": gold,
                    "is_correct": (pred == gold) if pred is not None else None,
                    "inference_time_sec": dt,
                    "model_output": gen,
                }
            )

        t1 = time.time()

        metrics = _diversity_metrics(preds)
        correct_frac = _correct_fraction(preds, gold)

        leader = metrics["leader"]
        max_frac = metrics["max_frac"]
        valid_n = metrics["valid_n"]

        if valid_n == 0:
            class_label = "invalid_all_none"
            leader_correct = None
        else:
            leader_correct = leader == gold

            if metrics["unanimous"]:
                class_label = "unanimous"
            elif max_frac >= 0.8:
                class_label = "lead80"
            elif max_frac >= 0.5:
                class_label = "lead50"
            else:
                class_label = "no_leader"

        self.last_full_record = {
            "index": index,
            "picked_index": picked_index,
            "id": example.get("id"),
            "question": example.get("question"),
            "options": {
                "A": example.get("opa"),
                "B": example.get("opb"),
                "C": example.get("opc"),
                "D": example.get("opd"),
            },
            "gold": gold,
            "subject_name": subject_name,
            "model_id": model_id,
            "model_slug": model_slug,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "n_branches": self.n_branches,
            "timestamp": _now_utc(),
            "branches": branch_records,
            "branch_preds": preds,
            "metrics": {
                **metrics,
                "correct_fraction": correct_frac,
                "leader_correct": leader_correct,
                "class": class_label,
                "wall_time_sec": (t1 - t0),
                "mean_branch_time_sec": sum(branch_times) / max(1, len(branch_times)),
            },
        }

        return {
            "index": index,
            "picked_index": picked_index,
            "id": example.get("id"),
            "gold": gold,
            "branch_preds": preds,
            "leader": leader,
            "max_frac": max_frac,
            "valid_n": valid_n,
            "none_n": metrics["none_n"],
            "variation_ratio": metrics["variation_ratio"],
            "entropy_bits": metrics["entropy_bits"],
            "correct_fraction": correct_frac,
            "leader_correct": leader_correct,
            "class": class_label,
            "subject_name": subject_name,
            "model_id": model_id,
            "model_slug": model_slug,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "n_branches": self.n_branches,
            "timestamp": _now_utc(),
        }
