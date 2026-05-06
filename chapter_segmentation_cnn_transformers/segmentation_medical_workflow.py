"""Utility scaffold for the segmentation chapter.

The commands in this file avoid heavyweight dependencies so they can run in the
repository smoke check. They prepare the reporting shape used by the chapter
homework. The actual optional SegFormer fine-tuning entry point is
``segformer_medical_finetune.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import tempfile
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MaskStats:
    path: str
    height: int
    width: int
    foreground_pixels: int
    total_pixels: int

    @property
    def foreground_fraction(self) -> float:
        if self.total_pixels == 0:
            return 0.0
        return self.foreground_pixels / self.total_pixels


def read_text_mask(path: Path) -> list[list[int]]:
    rows: list[list[int]] = []
    for line_no, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line:
            continue
        if "," in line:
            parts = [part.strip() for part in line.split(",")]
        elif " " in line:
            parts = [part.strip() for part in line.split()]
        else:
            parts = list(line)
        row: list[int] = []
        for part in parts:
            if part not in {"0", "1"}:
                raise ValueError(
                    f"{path}:{line_no}: mask value must be 0 or 1, got {part!r}"
                )
            row.append(int(part))
        if rows and len(row) != len(rows[0]):
            raise ValueError(f"{path}:{line_no}: inconsistent row width")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: mask is empty")
    return rows


def mask_stats(path: Path) -> MaskStats:
    mask = read_text_mask(path)
    height = len(mask)
    width = len(mask[0])
    foreground = sum(sum(row) for row in mask)
    return MaskStats(
        path=str(path),
        height=height,
        width=width,
        foreground_pixels=foreground,
        total_pixels=height * width,
    )


def iou_and_dice(
    true_values: Iterable[int], pred_values: Iterable[int]
) -> dict[str, float | int]:
    true = [int(value) for value in true_values]
    pred = [int(value) for value in pred_values]
    if len(true) != len(pred):
        raise ValueError("true and predicted masks must have the same number of pixels")
    intersection = sum(1 for t, p in zip(true, pred) if t == 1 and p == 1)
    true_count = sum(true)
    pred_count = sum(pred)
    union = true_count + pred_count - intersection
    iou = intersection / union if union else 1.0
    dice = (
        (2 * intersection) / (true_count + pred_count)
        if (true_count + pred_count)
        else 1.0
    )
    return {
        "intersection": intersection,
        "ground_truth_pixels": true_count,
        "predicted_pixels": pred_count,
        "union": union,
        "iou": iou,
        "dice": dice,
    }


def command_make_template(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    template = {
        "dataset": "Kvasir-SEG or another medical segmentation dataset",
        "columns": ["image_path", "mask_path", "split"],
        "splits": ["train", "validation", "test"],
        "notes": [
            "Keep the test split unused until the final selected recipe is fixed.",
            "Use nearest-neighbor interpolation for class-index masks.",
            "Record the seed if a random split is created.",
        ],
    }
    (output_dir / "segmentation_dataset_template.json").write_text(
        json.dumps(template, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "mask_path", "split"])
        writer.writeheader()
        writer.writerow(
            {
                "image_path": "images/example.png",
                "mask_path": "masks/example_mask.txt",
                "split": "train",
            }
        )
    print(f"Wrote segmentation template to {output_dir}")
    return 0


def command_metric_example(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true = [1] * 5 + [1] * 2 + [0] * 3
    pred = [1] * 5 + [0] * 2 + [1] * 3
    metrics = iou_and_dice(true, pred)
    (output_dir / "metric_example.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    return 0


def command_inspect_masks(args: argparse.Namespace) -> int:
    pairs_csv = Path(args.pairs_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(pairs_csv.open(encoding="utf-8")))
    if not rows:
        raise SystemExit("pairs CSV has no rows")
    stats: list[MaskStats] = []
    for row in rows:
        mask_path = Path(row["mask_path"])
        if not mask_path.is_absolute():
            mask_path = pairs_csv.parent / mask_path
        stats.append(mask_stats(mask_path))
    fractions = [item.foreground_fraction for item in stats]
    summary = {
        "mask_count": len(stats),
        "total_foreground_pixels": sum(item.foreground_pixels for item in stats),
        "total_pixels": sum(item.total_pixels for item in stats),
        "mean_foreground_fraction": statistics.fmean(fractions),
        "min_foreground_fraction": min(fractions),
        "max_foreground_fraction": max(fractions),
        "masks": [
            {
                "path": item.path,
                "height": item.height,
                "width": item.width,
                "foreground_pixels": item.foreground_pixels,
                "total_pixels": item.total_pixels,
                "foreground_fraction": item.foreground_fraction,
            }
            for item in stats
        ],
    }
    (output_dir / "mask_inspection.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Inspected {len(stats)} mask(s); mean foreground fraction {summary['mean_foreground_fraction']:.4f}"
    )
    return 0


def command_train_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "segformer_medical_finetune.py",
        "train",
        "--pairs-csv",
        str(Path(args.data_root) / "pairs.csv"),
        "--model-name",
        args.model,
        "--image-size",
        str(args.image_size),
        "--epochs",
        str(args.epochs),
        "--output-dir",
        str(output_dir),
    ]
    if args.dry_run:
        command.append("--dry-run")
    payload = {"dry_run": bool(args.dry_run), "command": command}
    (output_dir / "train_command.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(" ".join(command))
    return 0


def command_env_report(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ("torch", "transformers", "PIL", "numpy", "matplotlib"):
        packages[name] = importlib_util.find_spec(name) is not None
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages_available": packages,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote environment report to {output_dir}")
    return 0


def command_summarize_runs(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(runs_dir.glob("*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("run", path.parent.name)
        records.append(payload)
    if not records:
        raise SystemExit(f"No metrics.json files found under {runs_dir}")
    fieldnames = sorted({key for record in records for key in record})
    with (output_dir / "run_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "run_summary.json").write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Summarized {len(records)} run(s) into {output_dir}")
    return 0


def command_check_deps(args: argparse.Namespace) -> int:
    missing = [
        name
        for name in ("torch", "transformers", "PIL", "numpy")
        if importlib_util.find_spec(name) is None
    ]
    if missing and not args.allow_missing_deps:
        raise SystemExit(f"Missing optional dependencies: {', '.join(missing)}")
    if missing:
        print(f"Optional dependencies not installed: {', '.join(missing)}")
    else:
        print("Optional segmentation dependencies are available")
    return 0


def write_mask(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def command_smoke(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    write_mask(masks_dir / "mask_a.txt", ["0010", "0110", "0000", "0000"])
    write_mask(masks_dir / "mask_b.txt", ["0000", "0111", "0010", "0000"])
    pairs_csv = output_dir / "pairs.csv"
    with pairs_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "mask_path"])
        writer.writeheader()
        writer.writerow({"image_path": "image_a.png", "mask_path": "masks/mask_a.txt"})
        writer.writerow({"image_path": "image_b.png", "mask_path": "masks/mask_b.txt"})
    command_make_template(argparse.Namespace(output_dir=str(output_dir / "template")))
    command_metric_example(argparse.Namespace(output_dir=str(output_dir / "metrics")))
    command_inspect_masks(
        argparse.Namespace(
            pairs_csv=str(pairs_csv), output_dir=str(output_dir / "inspection")
        )
    )
    command_train_command(
        argparse.Namespace(
            data_root=str(output_dir),
            model="nvidia/segformer-b0-finetuned-ade-512-512",
            image_size=256,
            epochs=1,
            loss="bce-dice",
            output_dir=str(output_dir / "command"),
            dry_run=True,
        )
    )
    runs_dir = output_dir / "runs"
    (runs_dir / "quick").mkdir(parents=True, exist_ok=True)
    (runs_dir / "quick" / "metrics.json").write_text(
        json.dumps({"val_dice": 0.5, "val_iou": 0.333, "train_seconds": 1.0}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    command_summarize_runs(
        argparse.Namespace(
            runs_dir=str(runs_dir), output_dir=str(output_dir / "summary")
        )
    )
    command_env_report(argparse.Namespace(output_dir=str(output_dir / "env")))
    smoke_summary = {"status": "ok", "output_dir": str(output_dir)}
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(smoke_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Segmentation smoke check wrote artifacts to {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_template = subparsers.add_parser("make-template")
    make_template.add_argument("--output-dir", required=True)
    make_template.set_defaults(func=command_make_template)

    metric_example = subparsers.add_parser("metric-example")
    metric_example.add_argument("--output-dir", required=True)
    metric_example.set_defaults(func=command_metric_example)

    inspect_masks = subparsers.add_parser("inspect-masks")
    inspect_masks.add_argument("--pairs-csv", required=True)
    inspect_masks.add_argument("--output-dir", required=True)
    inspect_masks.set_defaults(func=command_inspect_masks)

    train_command = subparsers.add_parser("train-command")
    train_command.add_argument("--data-root", required=True)
    train_command.add_argument("--model", required=True)
    train_command.add_argument("--image-size", type=int, default=256)
    train_command.add_argument("--epochs", type=int, default=10)
    train_command.add_argument("--output-dir", required=True)
    train_command.add_argument("--dry-run", action="store_true")
    train_command.set_defaults(func=command_train_command)

    env_report = subparsers.add_parser("env-report")
    env_report.add_argument("--output-dir", required=True)
    env_report.set_defaults(func=command_env_report)

    summarize = subparsers.add_parser("summarize-runs")
    summarize.add_argument("--runs-dir", required=True)
    summarize.add_argument("--output-dir", required=True)
    summarize.set_defaults(func=command_summarize_runs)

    check_deps = subparsers.add_parser("check-deps")
    check_deps.add_argument("--allow-missing-deps", action="store_true")
    check_deps.set_defaults(func=command_check_deps)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument(
        "--output-dir",
        default=str(Path(tempfile.gettempdir()) / "adl-segmentation-smoke"),
    )
    smoke.set_defaults(func=command_smoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
