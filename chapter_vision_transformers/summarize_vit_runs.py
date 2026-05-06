#!/usr/bin/env python3
"""Collect pretrained ViT run artifacts into book-facing reference summaries."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = CODE_DIR / "artifacts" / "vit-reference"
CSV_LINETERMINATOR = "\n"


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path_text = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("run label must not be empty")
    path = Path(path_text).expanduser()
    return label, path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="Run directory as LABEL=PATH. Pass once per baseline/ablation/final run.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="vit-reference")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def metric(metrics: dict[str, Any], split: str, name: str) -> Any:
    payload = metrics.get(split, {})
    if isinstance(payload, dict):
        return payload.get(name, "")
    return ""


def run_row(label: str, run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"{metadata_path} missing")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"{metrics_path} missing")

    metadata = read_json(metadata_path)
    metrics = read_json(metrics_path)
    artifacts = metadata.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}

    return {
        "run_label": label,
        "run_dir": str(run_dir),
        "dataset": metadata.get("dataset", ""),
        "dataset_source": metadata.get("dataset_source", ""),
        "split_rule": metadata.get("split_rule", ""),
        "checkpoint": metadata.get("checkpoint", ""),
        "image_size": metadata.get("preprocessing", {}).get("image_size", ""),
        "trainable_mode": metadata.get("trainable_mode", ""),
        "lora_config": metadata.get("lora", ""),
        "total_parameters": metadata.get("total_parameters", ""),
        "trainable_parameters": metadata.get("trainable_parameters", ""),
        "optimizer": metadata.get("optimizer", ""),
        "learning_rate": metadata.get("learning_rate", ""),
        "weight_decay": metadata.get("weight_decay", ""),
        "batch_size": metadata.get("batch_size", ""),
        "gradient_accumulation_steps": metadata.get("gradient_accumulation_steps", ""),
        "precision": metadata.get("precision", ""),
        "hardware": metadata.get("hardware", ""),
        "validation_accuracy": metric(metrics, "validation", "accuracy"),
        "validation_loss": metric(metrics, "validation", "loss"),
        "final_test_accuracy": metric(metrics, "test", "accuracy"),
        "elapsed_seconds": metrics.get("elapsed_seconds", ""),
        "training_examples_per_second": metrics.get("training_examples_per_second", ""),
        "peak_memory": metrics.get("peak_memory", ""),
        "validation_mistake_table": artifacts.get("validation_prediction_examples", ""),
        "command": metadata.get("command", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator=CSV_LINETERMINATOR)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "Vision Transformers reference run summary",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "Host: not-recorded",
        f"Platform: {platform.platform()}",
        f"Rows: {len(rows)}",
        "",
        "Source runs:",
    ]
    for row in rows:
        lines.append(f"- {row['run_label']}: {row['run_dir']}")
    lines.extend(
        [
            "",
            "Use this summary only after validating that all listed runs used the same dataset split.",
            "The final test accuracy field should be filled only for the selected final recipe.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = [run_row(label, path) for label, path in args.run]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"{args.prefix}-results.csv"
    metadata_path = args.output_dir / f"{args.prefix}-results-metadata.txt"
    write_csv(results_path, rows)
    write_metadata(metadata_path, rows)
    print(results_path)
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
