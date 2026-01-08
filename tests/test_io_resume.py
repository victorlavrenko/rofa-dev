from rofa.io import load_progress, write_progress


def test_progress_roundtrip(tmp_path):
    progress_path = tmp_path / "progress.json"
    payload = {
        "run_id": "test-run",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "i": 10,
        "picked": 5,
        "subject_counts": {"Cardiology": 3},
        "summary_written": 5,
        "full_written": 2,
    }

    write_progress(str(progress_path), payload)
    loaded = load_progress(str(progress_path))

    assert loaded == payload
