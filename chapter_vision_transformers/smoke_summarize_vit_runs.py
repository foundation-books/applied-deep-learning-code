#!/usr/bin/env python3
"""Smoke test the ViT run summarizer on representative fake artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile

from summarize_vit_runs import main


def write_run(
    run_dir: Path,
    *,
    trainable_mode: str,
    validation_accuracy: float,
    test_accuracy: float | None = None,
) -> None:
    run_dir.mkdir(parents=True)
    metadata = {
        "command": f"python pretrained_vit_experiment.py --trainable-mode {trainable_mode}",
        "dataset": "food101",
        "dataset_source": "TorchVision Food-101 under data",
        "split_rule": "fixed smoke split",
        "checkpoint": "google/vit-base-patch16-224-in21k",
        "preprocessing": {"image_size": 224},
        "trainable_mode": trainable_mode,
        "lora": (
            {
                "r": 16,
                "alpha": 16,
                "dropout": 0.1,
                "target_modules": ["query", "value"],
                "modules_to_save": ["classifier"],
            }
            if trainable_mode == "lora"
            else None
        ),
        "total_parameters": 85806346,
        "trainable_parameters": 7680,
        "optimizer": "AdamW",
        "learning_rate": 0.0002,
        "weight_decay": 0.01,
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "precision": "bf16",
        "hardware": "CUDA reference GPU",
        "artifacts": {
            "validation_prediction_examples": str(run_dir / "validation_prediction_examples.csv")
        },
    }
    metrics: dict[str, object] = {
        "validation": {"accuracy": validation_accuracy, "loss": 0.75},
        "test": {"status": "not requested"},
        "elapsed_seconds": 123.4,
        "training_examples_per_second": 25.6,
        "peak_memory": "4.250 GB CUDA max_memory_allocated",
    }
    if test_accuracy is not None:
        metrics["test"] = {"accuracy": test_accuracy, "loss": 0.72}

    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "validation_prediction_examples.csv").write_text(
        "source,true_label,predicted_label,confidence,error_note\n",
        encoding="utf-8",
    )


def run_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp_text:
        tmp = Path(tmp_text)
        baseline = tmp / "baseline"
        ablation = tmp / "ablation"
        out_dir = tmp / "figures"
        write_run(baseline, trainable_mode="head", validation_accuracy=0.81)
        write_run(ablation, trainable_mode="lora", validation_accuracy=0.84)

        exit_code = main(
            [
                "--run",
                f"baseline={baseline}",
                "--run",
                f"ablation={ablation}",
                "--output-dir",
                str(out_dir),
                "--prefix",
                "smoke-vit",
            ]
        )
        if exit_code != 0:
            raise AssertionError(f"summarizer returned {exit_code}")

        results_path = out_dir / "smoke-vit-results.csv"
        metadata_path = out_dir / "smoke-vit-results-metadata.txt"
        rows = list(csv.DictReader(results_path.open(encoding="utf-8")))
        if [row["run_label"] for row in rows] != ["baseline", "ablation"]:
            raise AssertionError("unexpected run labels in summary CSV")
        if rows[0]["trainable_mode"] != "head":
            raise AssertionError("baseline trainable mode was not preserved")
        if rows[1]["validation_accuracy"] != "0.84":
            raise AssertionError("ablation validation accuracy was not preserved")
        if "Source runs:" not in metadata_path.read_text(encoding="utf-8"):
            raise AssertionError("summary metadata omitted source runs")


if __name__ == "__main__":
    run_smoke()
