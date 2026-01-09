import json
import zipfile

from rofa.core.io import load_progress, unpack_zip


def test_load_progress_missing_returns_none(tmp_path):
    assert load_progress(str(tmp_path / "missing.json")) is None


def test_unpack_zip_extracts_files(tmp_path):
    zip_path = tmp_path / "sample.zip"
    extract_dir = tmp_path / "extracted"
    payload = {"value": 42}

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data.json", json.dumps(payload))

    unpack_zip(str(zip_path), str(extract_dir))
    extracted_file = extract_dir / "data.json"

    assert extracted_file.exists()
    assert json.loads(extracted_file.read_text()) == payload
