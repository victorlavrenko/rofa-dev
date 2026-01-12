import json
import sys
from pathlib import Path

import pandas as pd

from rofa.papers.from_answers_to_hypotheses import analysis, notebook_helpers
from scripts import analyze

FIXTURE_ROOT = Path(__file__).resolve().parent / "data"
FIXTURE_RUN_DIR = FIXTURE_ROOT / "k_sample_ensemble_test"
FIXTURE_GREEDY_DIR = FIXTURE_ROOT / "greedy_test"
FIXTURE_REPORT_DIR = FIXTURE_ROOT / "reports" / "k_sample_ensemble_test"


def _load_fixture_json(name: str) -> dict:
    return json.loads((FIXTURE_REPORT_DIR / name).read_text(encoding="utf-8"))


def test_run_report_matches_fixture() -> None:
    df_summary = analysis.load_summary(str(FIXTURE_RUN_DIR))
    report = analysis.run_report(df_summary)
    expected = _load_fixture_json("report.json")
    assert report == expected


def test_cli_analyze_matches_fixture(tmp_path: Path) -> None:
    output_path = tmp_path / "report.json"
    argv_backup = sys.argv
    try:
        sys.argv = [
            "analyze.py",
            "--run",
            str(FIXTURE_RUN_DIR),
            "--output",
            str(output_path),
        ]
        analyze.main()
    finally:
        sys.argv = argv_backup

    report = json.loads(output_path.read_text(encoding="utf-8"))
    expected = _load_fixture_json("report.json")
    assert report == expected


def test_notebook_reports_match_fixtures(tmp_path: Path, monkeypatch) -> None:
    df_greedy = analysis.load_summary(str(FIXTURE_GREEDY_DIR))
    df_branches = analysis.load_summary(str(FIXTURE_RUN_DIR))

    df_greedy_accuracy = pd.DataFrame({"value": [analysis.accuracy_greedy(df_greedy)]})
    df_leader_accuracy = pd.DataFrame({"value": [analysis.accuracy_leader(df_branches)]})
    unanimous_stats = analysis.unanimous_stats(df_branches)
    near_unanimous_stats = analysis.near_unanimous_stats(df_branches)
    df_top2 = pd.DataFrame({"value": [analysis.top2_coverage(df_branches)]})
    df_max_frac = (
        analysis.max_frac_distribution(df_branches)
        .rename_axis("max_frac_bin")
        .reset_index(name="count")
    )
    df_rw_other = analysis.rw_other_breakdown(df_branches).reindex(
        columns=["Other", "R", "W"]
    )
    df_subject_breakdown = notebook_helpers.subject_breakdown(df_greedy, df_branches)

    metadata = {"resolved_runs": {"k_sample_ensemble": str(FIXTURE_RUN_DIR)}}

    monkeypatch.chdir(tmp_path)
    report_dir = notebook_helpers.export_paper_reports(
        metadata,
        df_greedy_accuracy,
        df_leader_accuracy,
        unanimous_stats,
        near_unanimous_stats,
        df_top2,
        df_max_frac,
        df_rw_other,
        df_subject_breakdown,
    )

    expected_report = _load_fixture_json("paper_report.json")
    generated_report = json.loads((report_dir / "paper_report.json").read_text())
    assert generated_report == expected_report

    for filename in [
        "max_frac_distribution.csv",
        "rw_other_breakdown.csv",
        "subject_accuracy.csv",
    ]:
        expected_text = (FIXTURE_REPORT_DIR / filename).read_text(encoding="utf-8")
        generated_text = (report_dir / filename).read_text(encoding="utf-8")
        assert generated_text == expected_text
