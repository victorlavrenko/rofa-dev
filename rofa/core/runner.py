"""Shared dataset loop for generation runs."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Optional

from tqdm import tqdm

from .io import _append_jsonl, _now_utc, load_manifest, load_progress, write_manifest, write_progress
from .parse import cop_to_letter
from .question_set import (
    create_question_set,
    expand_question_set,
    load_filtered_dataset,
    load_question_set,
    question_hash,
    save_question_set,
)
from .schemas import GenerationConfig, RunConfig, RunManifest


def _resolve_example(ds, entry: Dict[str, Any], id_index_cache: Dict[str, int]) -> Dict[str, Any]:
    """Resolve a question set entry to a dataset example."""
    dataset_index = entry.get("dataset_index")
    if dataset_index is not None and 0 <= dataset_index < len(ds):
        return ds[dataset_index]

    example_id = entry.get("id")
    if example_id is None:
        raise ValueError("Question set entry missing dataset_index and id.")

    if example_id in id_index_cache:
        return ds[id_index_cache[example_id]]

    for idx, ex in enumerate(ds):
        if ex.get("id") == example_id:
            id_index_cache[example_id] = idx
            return ex

    raise ValueError(f"Could not resolve example id {example_id}.")


def _update_progress_bar(
    bar: tqdm,
    *,
    completed: int,
    total: int,
    start_time: float,
    heartbeat: Optional[str] = None,
) -> None:
    """Update the tqdm progress bar with elapsed time and ETA."""
    elapsed = time.time() - start_time
    avg = elapsed / max(1, completed)
    remaining = max(0.0, (total - completed) * avg)
    ex_per_sec = completed / max(1e-9, elapsed)
    bar.set_postfix(
        {
            "elapsed_s": f"{elapsed:0.1f}",
            "eta_s": f"{remaining:0.1f}",
            "ex/s": f"{ex_per_sec:0.2f}",
        }
    )
    if heartbeat:
        bar.write(heartbeat)


def _resolve_run_paths(config: GenerationConfig) -> Dict[str, str]:
    if config.run_id:
        run_id = config.run_id
        run_dir = os.path.join(config.out_dir, run_id)
    else:
        run_dir = config.out_dir
        run_id = os.path.basename(os.path.abspath(run_dir)) or uuid.uuid4().hex
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "progress_path": os.path.join(run_dir, "progress.json"),
        "manifest_path": os.path.join(run_dir, "manifest.json"),
        "summary_path": os.path.join(run_dir, "summary.jsonl"),
        "full_path": os.path.join(run_dir, "full.jsonl"),
        "question_set_path": os.path.join(run_dir, "question_set.json"),
    }


def _validate_resume(
    *,
    progress: Optional[Dict[str, Any]],
    resume: bool,
    run_id: str,
    summary_path: str,
) -> None:
    if progress:
        progress_run_id = progress.get("run_id")
        if progress_run_id and progress_run_id != run_id:
            raise ValueError(
                f"Run ID mismatch: progress has {progress_run_id} but run_id is {run_id}."
            )
        if not resume:
            raise ValueError(
                "progress.json exists but resume was disabled; refuse to overwrite artifacts."
            )
    elif resume and os.path.exists(summary_path):
        raise ValueError(
            "summary.jsonl exists but no progress.json was found; cannot safely resume."
        )


def _load_or_create_question_set(
    config: GenerationConfig,
    *,
    question_set_path: str,
    selection_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, str],
):
    if os.path.exists(question_set_path):
        qs = load_question_set(question_set_path)
    elif config.question_set_path:
        qs = load_question_set(config.question_set_path)
        save_question_set(qs, question_set_path)
    else:
        qs = create_question_set(dataset_cfg, selection_cfg)
        save_question_set(qs, question_set_path)
    if config.expand:
        if qs.selection.get("seed") != selection_cfg["seed"]:
            raise ValueError("Seed mismatch between config and question set.")
        if qs.selection.get("subjects") != selection_cfg["subjects"]:
            raise ValueError("Subjects mismatch between config and question set.")
        expand_cfg = dict(selection_cfg)
        expand_cfg["max_per_subject"] = qs.selection.get(
            "max_per_subject", selection_cfg["max_per_subject"]
        )
        qs = expand_question_set(qs, dataset_cfg, expand_cfg)
        save_question_set(qs, question_set_path)
    if config.question_set_path and os.path.exists(question_set_path):
        loaded_qs = load_question_set(question_set_path)
        if loaded_qs.qs_id != qs.qs_id:
            raise ValueError("Question set mismatch between run directory and provided path.")
    return qs


def _ensure_manifest(
    config: GenerationConfig,
    *,
    run_id: str,
    manifest_path: str,
    question_set_id: str,
    selection_cfg: Dict[str, Any],
) -> None:
    if os.path.exists(manifest_path):
        if not config.expand:
            return
        manifest = load_manifest(manifest_path)
        if manifest is None:
            return
        updated_config = RunConfig(
            method=config.method,
            model_id=config.model_id,
            seed=config.seed,
            max_new_tokens=config.max_new_tokens,
            n=config.n,
            subjects=config.subjects,
            max_per_subject=selection_cfg["max_per_subject"],
            dataset_name=config.dataset_name,
            dataset_split=config.dataset_split,
            n_branches=config.n_branches,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            question_set_id=question_set_id,
        )
        updated_manifest = RunManifest(
            run_id=manifest.run_id,
            created_at=manifest.created_at,
            method=config.method,
            config=updated_config,
            notes=manifest.notes,
        )
        write_manifest(manifest_path, updated_manifest)
        return
    run_config = RunConfig(
        method=config.method,
        model_id=config.model_id,
        seed=config.seed,
        max_new_tokens=config.max_new_tokens,
        n=config.n,
        subjects=config.subjects,
        max_per_subject=selection_cfg["max_per_subject"],
        dataset_name=config.dataset_name,
        dataset_split=config.dataset_split,
        n_branches=config.n_branches,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        question_set_id=question_set_id,
    )
    manifest = RunManifest(
        run_id=run_id,
        created_at=_now_utc(),
        method=config.method,
        config=run_config,
    )
    write_manifest(manifest_path, manifest)


def _init_progress_state(
    *,
    progress: Optional[Dict[str, Any]],
    summary_path: str,
    full_path: str,
    write_full_records: bool,
) -> Dict[str, int]:
    if progress:
        return {
            "position": progress.get("position", 0),
            "summary_written": progress.get("summary_written", 0),
            "full_written": progress.get("full_written", 0),
        }
    open(summary_path, "w", encoding="utf-8").close()
    if write_full_records:
        open(full_path, "w", encoding="utf-8").close()
    return {"position": 0, "summary_written": 0, "full_written": 0}


def run_generation(config: GenerationConfig) -> Dict[str, Any]:
    """Run a generation job based on a question set.

    Args:
        config: GenerationConfig containing model handles, method implementation, and
            output locations.

    Returns:
        Summary information for the completed run segment.

    Artifacts:
        Writes ``manifest.json``, ``question_set.json``, ``progress.json``, and
        ``summary.jsonl`` to the run directory. Writes ``full.jsonl`` for ensemble runs
        when ``write_full_records=True``.

    Raises:
        ValueError: If required model components are missing, if resume state is
            inconsistent with on-disk artifacts, or if the question set mismatches the
            run directory.

    Notes:
        The run artifact schema is stable across papers. Paper-specific logic lives in
        ``config.method_impl`` (e.g., prompts or decoding strategy).
    """
    if config.tokenizer is None or config.model is None or config.method_impl is None:
        raise ValueError("GenerationConfig must include tokenizer, model, and method_impl.")
    paths = _resolve_run_paths(config)
    run_id = paths["run_id"]
    run_dir = paths["run_dir"]
    progress_path = paths["progress_path"]
    manifest_path = paths["manifest_path"]
    summary_path = paths["summary_path"]
    full_path = paths["full_path"]
    question_set_path = paths["question_set_path"]

    os.makedirs(run_dir, exist_ok=True)

    progress = load_progress(progress_path)
    resume = config.resume if config.resume is not None else bool(progress)
    _validate_resume(progress=progress, resume=resume, run_id=run_id, summary_path=summary_path)

    dataset_cfg = {
        "dataset_name": config.dataset_name,
        "dataset_split": config.dataset_split,
    }
    selection_cfg = {
        "seed": config.seed,
        "n": config.n,
        "subjects": config.subjects,
        "max_per_subject": config.n / config.subjects * 1.1 + 1,
    }

    qs = _load_or_create_question_set(
        config,
        question_set_path=question_set_path,
        selection_cfg=selection_cfg,
        dataset_cfg=dataset_cfg,
    )
    if config.expand:
        selection_cfg["max_per_subject"] = qs.selection.get(
            "max_per_subject", selection_cfg["max_per_subject"]
        )
    _ensure_manifest(
        config,
        run_id=run_id,
        manifest_path=manifest_path,
        question_set_id=qs.qs_id,
        selection_cfg=selection_cfg,
    )

    ds = load_filtered_dataset(config.dataset_name, config.dataset_split)

    progress_state = _init_progress_state(
        progress=progress,
        summary_path=summary_path,
        full_path=full_path,
        write_full_records=config.write_full_records,
    )
    position = progress_state["position"]
    summary_written = progress_state["summary_written"]
    full_written = progress_state["full_written"]

    total = len(qs.examples)
    segment_total = total - position
    bar = None
    start_time = time.time()
    if config.progress:
        bar = tqdm(total=total, initial=position, dynamic_ncols=True)

    id_index_cache: Dict[str, int] = {}
    completed_since_start = 0

    for idx in range(position, total):
        entry = qs.examples[idx]
        ex = _resolve_example(ds, entry, id_index_cache)
        entry_hash = entry.get("question_hash")
        if entry_hash and question_hash(ex) != entry_hash:
            print(f"  Warning: question hash mismatch at index {idx}.")
        subj = ex.get("subject_name", "Unknown") or "Unknown"

        context = {
            "tokenizer": config.tokenizer,
            "model": config.model,
            "max_new_tokens": config.max_new_tokens,
            "seed": config.seed,
            "index": idx,
            "picked_index": idx + 1,
            "subject_name": subj,
            "gold": cop_to_letter(ex["cop"]),
        }

        record = config.method_impl.run_one(ex, context)
        _append_jsonl(summary_path, record)
        summary_written += 1

        full_record = getattr(config.method_impl, "last_full_record", None)
        if config.write_full_records and full_record is not None:
            _append_jsonl(full_path, full_record)
            full_written += 1

        if record.get("prediction") is None and "prediction" in record:
            print("  Warning: could not extract answer from model output.")

        position = idx + 1
        completed_since_start += 1

        write_progress(
            progress_path,
            {
                "run_id": run_id,
                "timestamp": _now_utc(),
                "position": position,
                "summary_written": summary_written,
                "full_written": full_written,
            },
        )

        if bar:
            bar.update(1)
            heartbeat = None
            if config.heartbeat_every and completed_since_start % config.heartbeat_every == 0:
                heartbeat = f"heartbeat: summary_written={summary_written}"
            _update_progress_bar(
                bar,
                completed=completed_since_start,
                total=segment_total,
                start_time=start_time,
                heartbeat=heartbeat,
            )

    if bar:
        bar.close()

    return {
        "summary_written": summary_written,
        "full_written": full_written,
        "position": position,
    }
