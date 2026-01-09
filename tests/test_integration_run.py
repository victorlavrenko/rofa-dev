import json
import sys
from pathlib import Path

import pandas as pd

from rofa.papers.from_answers_to_hypotheses import analysis as paper_analysis
from scripts import validate_run


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_summary(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(record) for record in records]
    path.write_text("\n".join(lines) + "\n")


def _make_manifest(run_id: str, method: str) -> dict:
    return {
        "run_id": run_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "method": method,
        "config": {
            "method": method,
            "model_id": "test-model",
            "seed": 42,
            "max_new_tokens": 64,
            "n": 2,
            "subjects": 1,
            "max_per_subject": 2,
            "dataset_name": "dummy",
            "dataset_split": "validation",
            "question_set_id": "qs_test",
        },
    }


def _make_progress(run_id: str, summary_written: int) -> dict:
    return {
        "run_id": run_id,
        "timestamp": "2026-01-01T00:00:01+00:00",
        "position": summary_written,
        "summary_written": summary_written,
        "full_written": 0,
    }


def _make_question_set() -> dict:
    return {
        "qs_id": "qs_test",
        "dataset_name": "dummy",
        "dataset_split": "validation",
        "dataset_revision": None,
        "dataset_fingerprint": None,
        "selection": {"seed": 42, "n": 2, "subjects": 1, "max_per_subject": 2},
        "examples": [
            {"dataset_index": 0, "id": "ex0", "question_hash": "hash0"},
            {"dataset_index": 1, "id": "ex1", "question_hash": "hash1"},
        ],
    }


def _write_run(run_dir: Path, method: str, summary_records: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "manifest.json", _make_manifest(run_dir.name, method))
    _write_json(run_dir / "progress.json", _make_progress(run_dir.name, len(summary_records)))
    _write_json(run_dir / "question_set.json", _make_question_set())
    _write_summary(run_dir / "summary.jsonl", summary_records)


def test_integration_validate_and_metrics(tmp_path: Path) -> None:
    run_dir_greedy = tmp_path / "greedy_run"
    run_dir_k_sample = tmp_path / "k_sample_run"

    greedy_records = [
        {
            "index": 0,
            "id": "ex0",
            "gold": "A",
            "prediction": "A",
            "is_correct": True,
            "model_output": "Final answer: A",
            "subject_name": "Test",
            "timestamp": "2026-01-01T00:00:02+00:00",
        },
        {
            "index": 1,
            "id": "ex1",
            "gold": "B",
            "prediction": "A",
            "is_correct": False,
            "model_output": "Final answer: A",
            "subject_name": "Test",
            "timestamp": "2026-01-01T00:00:03+00:00",
        },
    ]
    k_sample_records = [
        {
            "index": 0,
            "picked_index": 1,
            "id": "ex0",
            "gold": "A",
            "branch_preds": ["A", "A", "B"],
            "leader": "A",
            "max_frac": 2 / 3,
            "valid_n": 3,
            "none_n": 0,
            "variation_ratio": 1 / 3,
            "entropy_bits": 0.918,
            "correct_fraction": 2 / 3,
            "leader_correct": True,
            "class": "lead50",
            "subject_name": "Test",
            "timestamp": "2026-01-01T00:00:04+00:00",
        }
    ]

    _write_run(run_dir_greedy, "greedy", greedy_records)
    _write_run(run_dir_k_sample, "k_sample_ensemble", k_sample_records)

    argv_backup = sys.argv
    try:
        sys.argv = ["validate_run.py", "--run", str(run_dir_greedy)]
        validate_run.main()
        sys.argv = ["validate_run.py", "--run", str(run_dir_k_sample)]
        validate_run.main()
    finally:
        sys.argv = argv_backup

    df_greedy, df_branches, _ = paper_analysis.load_paper_runs(
        [str(run_dir_greedy), str(run_dir_k_sample)]
    )
    df_accuracy = paper_analysis.compute_table_accuracy(df_greedy, df_branches)
    df_top2 = paper_analysis.compute_table_top2(df_branches)

    assert isinstance(df_accuracy, pd.DataFrame)
    assert df_accuracy.loc[df_accuracy["metric"] == "greedy_accuracy", "value"].iloc[0] == 0.5
    assert df_top2.loc[df_top2["metric"] == "top2_coverage", "value"].iloc[0] == 1.0
