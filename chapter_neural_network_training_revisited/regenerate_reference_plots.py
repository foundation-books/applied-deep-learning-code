#!/usr/bin/env python3
"""Regenerate chapter reference plots from saved experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

import transformer_distillation_experiment as experiment


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "reference_artifacts"

PLOT_FILES = {
    "training": "nntrev-reference-training-curves.png",
    "confusion": "nntrev-reference-confusion-matrix.png",
    "tradeoff": "nntrev-reference-accuracy-latency-size.png",
    "calibration": "nntrev-reference-calibration-ece.png",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing the nntrev-reference CSV artifacts and plot PNGs.",
    )
    parser.add_argument(
        "--plot",
        action="append",
        choices=sorted(PLOT_FILES),
        help="Plot to regenerate. May be repeated. Defaults to all plots.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare regenerated plots with committed PNGs without overwriting them.",
    )
    return parser.parse_args(argv)


def read_rows(artifact_dir: Path, filename: str) -> list[dict[str, str]]:
    path = artifact_dir / filename
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rows_for(selected: set[str], plot: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return rows if plot in selected else []


def compare_or_copy(source: Path, destination: Path, *, check: bool) -> bool:
    if check:
        if source.read_bytes() == destination.read_bytes():
            print(f"OK {destination.relative_to(REPO_ROOT)}")
            return True
        print(f"DIFF {destination.relative_to(REPO_ROOT)}", file=sys.stderr)
        return False

    shutil.copy2(source, destination)
    print(f"Wrote {destination.relative_to(REPO_ROOT)}")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected = set(args.plot or PLOT_FILES)
    artifact_dir = args.artifact_dir.resolve()

    plotnine = experiment.load_plotnine_dependencies()
    if plotnine is None:
        print("Missing plotting dependencies. Install matplotlib, pandas, and plotnine.", file=sys.stderr)
        return 1

    result_rows = read_rows(artifact_dir, "nntrev-reference-distillation-results.csv")
    history = rows_for(selected, "training", read_rows(artifact_dir, "nntrev-reference-training-curves.csv"))
    confusion = rows_for(selected, "confusion", read_rows(artifact_dir, "nntrev-reference-confusion-matrix.csv"))
    latency = rows_for(selected, "tradeoff", read_rows(artifact_dir, "nntrev-reference-latency-table.csv"))
    calibration = rows_for(selected, "calibration", read_rows(artifact_dir, "nntrev-reference-calibration-summary.csv"))

    with tempfile.TemporaryDirectory(prefix="nntrev-plots-") as tmp:
        output_dir = Path(tmp)
        experiment.save_plots(
            output_dir,
            history,
            confusion,
            result_rows,
            latency,
            calibration,
            {"plotnine": plotnine},
        )

        ok = True
        for plot_name in sorted(selected):
            filename = PLOT_FILES[plot_name]
            source = output_dir / filename
            if not source.exists():
                print(f"Did not generate {filename}", file=sys.stderr)
                ok = False
                continue
            ok = compare_or_copy(source, artifact_dir / filename, check=args.check) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
