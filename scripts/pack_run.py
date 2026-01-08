"""Package a ROFA run directory into a zip archive."""

from __future__ import annotations

import argparse
import os
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zip a ROFA run directory for release.")
    parser.add_argument("--run-dir", required=True, help="Path to a run directory.")
    parser.add_argument(
        "--output",
        help="Optional output zip path (defaults to <run-dir>.zip).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise ValueError(f"Run directory not found: {run_dir}")

    output_path = args.output or f"{run_dir}.zip"
    base_dir = os.path.dirname(run_dir)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(run_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, base_dir)
                zf.write(full_path, rel_path)

    print(f"Wrote archive: {output_path}")


if __name__ == "__main__":
    main()
