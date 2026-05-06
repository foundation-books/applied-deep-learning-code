#!/usr/bin/env python3
"""Fine-tune a Transformer classifier for toxic-comment length comparisons.

The Jigsaw toxic-comment data is not bundled with the repository. Download the
Kaggle training CSV and place it at ``chapter_attention_transformers/data/train.csv``,
or pass ``--data-dir`` / ``--train-csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import numbers
from pathlib import Path
import platform
import sys
import time
from typing import Any


LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory containing Kaggle train.csv.")
    parser.add_argument("--train-csv", default="train.csv", help="Training CSV name inside --data-dir.")
    parser.add_argument("--text-column", default="comment_text")
    parser.add_argument("--checkpoint", default="distilbert-base-uncased")
    parser.add_argument("--max-lengths", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument(
        "--mixed-precision",
        choices=("none", "fp16", "bf16"),
        default="none",
        help="Use fp16 or bf16 only on supported CUDA hardware.",
    )
    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        help="Use a small generated multi-label dataset instead of Kaggle files.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a tiny configuration for dependency and pipeline checks.",
    )
    parser.add_argument("--artifact-dir", default="runs/toxic-length-comparison")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--error-example-limit", type=int, default=20)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require tokenizer/checkpoint files to already be cached locally.",
    )
    parser.add_argument("--check-deps", action="store_true", help="Import dependencies and exit.")
    parser.add_argument(
        "--allow-missing-deps",
        action="store_true",
        help="Return success when dependency imports fail; useful for repository checks.",
    )
    return parser.parse_args()


def load_dependencies(args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        import numpy as np
        import pandas as pd
        import torch
        from sklearn.metrics import f1_score, roc_auc_score
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except Exception as exc:  # pragma: no cover - depends on local environment
        message = f"Missing or unusable dependency: {exc}"
        if args.allow_missing_deps:
            print(message)
            print("Dependency check skipped because --allow-missing-deps was supplied.")
            return None
        raise RuntimeError(message) from exc

    return {
        "np": np,
        "pd": pd,
        "torch": torch,
        "f1_score": f1_score,
        "roc_auc_score": roc_auc_score,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def apply_quick_settings(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.synthetic_data = True
    args.checkpoint = "hf-internal-testing/tiny-random-distilbert"
    args.max_lengths = [16, 32]
    args.epochs = 1.0
    args.batch_size = min(args.batch_size, 4)
    args.eval_batch_size = min(args.eval_batch_size, 8)
    args.limit_train = args.limit_train or 48
    args.limit_val = args.limit_val or 24


def load_kaggle_frame(args: argparse.Namespace, pd: Any) -> Any:
    path = Path(args.data_dir) / args.train_csv
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Jigsaw toxic-comment train.csv from Kaggle "
            "or pass --synthetic-data for a quick pipeline check."
        )
    frame = pd.read_csv(path)
    required = {args.text_column, *LABEL_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame[[args.text_column, *LABEL_COLUMNS]].copy()


def make_synthetic_frame(args: argparse.Namespace, pd: Any) -> Any:
    rows: list[dict[str, Any]] = []
    templates = [
        ("thanks for the careful explanation", [0, 0, 0, 0, 0, 0]),
        ("this is toxic and insulting", [1, 0, 0, 0, 1, 0]),
        ("obscene toxic insult in a long argument", [1, 0, 1, 0, 1, 0]),
        ("a direct threat appears in the message", [1, 0, 0, 1, 0, 0]),
        ("identity hate and obscene abuse", [1, 0, 1, 0, 0, 1]),
        ("severe toxic attack with threat and insult", [1, 1, 0, 1, 1, 0]),
        ("ordinary disagreement about the topic", [0, 0, 0, 0, 0, 0]),
        ("neutral reference to identity without abuse", [0, 0, 0, 0, 0, 0]),
    ]
    repeats = max(12, math.ceil(((args.limit_train or 48) + (args.limit_val or 24)) / len(templates)))
    for repeat in range(repeats):
        for text, labels in templates:
            row = {args.text_column: f"{text} example {repeat}"}
            row.update(dict(zip(LABEL_COLUMNS, labels)))
            rows.append(row)
    return pd.DataFrame(rows)


def split_frame(args: argparse.Namespace, frame: Any, np: Any) -> tuple[Any, Any]:
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1.")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(frame))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * args.validation_fraction)))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    if args.limit_train is not None:
        train_indices = train_indices[: args.limit_train]
    if args.limit_val is not None:
        val_indices = val_indices[: args.limit_val]

    return frame.iloc[train_indices].reset_index(drop=True), frame.iloc[val_indices].reset_index(drop=True)


class CommentDataset:
    def __init__(self, encodings: dict[str, Any], labels: Any, torch: Any) -> None:
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.torch = torch

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: self.torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def make_dataset(frame: Any, tokenizer: Any, args: argparse.Namespace, max_length: int, np: Any, torch: Any) -> Any:
    texts = frame[args.text_column].astype(str).tolist()
    labels = frame[LABEL_COLUMNS].to_numpy(dtype=np.float32)
    encodings = tokenizer(texts, truncation=True, max_length=max_length)
    return CommentDataset(encodings, labels, torch)


def sigmoid(x: Any, np: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-x))


def build_metric_fn(np: Any, f1_score: Any, roc_auc_score: Any):
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        probabilities = sigmoid(logits, np)
        predictions = (probabilities >= 0.5).astype(int)
        metrics = {
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        }
        try:
            metrics["macro_roc_auc"] = float(roc_auc_score(labels, probabilities, average="macro"))
        except ValueError:
            metrics["macro_roc_auc"] = float("nan")
        return metrics

    return compute_metrics


def training_arguments_kwargs(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "save_strategy": "no",
        "logging_strategy": "epoch",
        "report_to": [],
        "seed": args.seed,
        "fp16": args.mixed_precision == "fp16",
        "bf16": args.mixed_precision == "bf16",
    }
    return kwargs


def make_training_arguments(TrainingArguments: Any, args: argparse.Namespace, output_dir: Path) -> Any:
    kwargs = training_arguments_kwargs(args, output_dir)
    try:
        return TrainingArguments(eval_strategy="epoch", **kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **kwargs)


def format_hms(seconds: float) -> str:
    whole = int(round(seconds))
    minutes, sec = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def package_versions(deps: dict[str, Any]) -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("np", "pd", "torch"):
        module = deps[name]
        versions[name] = str(getattr(module, "__version__", "unknown"))
    import sklearn
    import transformers

    versions["sklearn"] = str(getattr(sklearn, "__version__", "unknown"))
    versions["transformers"] = str(getattr(transformers, "__version__", "unknown"))
    return versions


class EpochTimerFactory:
    def __init__(self, TrainerCallback: Any) -> None:
        self.TrainerCallback = TrainerCallback

    def make(self) -> tuple[Any, list[float]]:
        epoch_seconds: list[float] = []
        callback_base = self.TrainerCallback

        class EpochTimer(callback_base):
            def __init__(self) -> None:
                self.started_at: float | None = None

            def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                self.started_at = time.perf_counter()

            def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                if self.started_at is not None:
                    epoch_seconds.append(time.perf_counter() - self.started_at)

        return EpochTimer(), epoch_seconds


def count_trainable_parameters(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def save_error_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "validation_index",
        "label",
        "probability",
        "confidence",
        "true_value",
        "predicted_value",
        "comment_text",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_error_examples(
    setting_dir: Path,
    val_frame: Any,
    args: argparse.Namespace,
    logits: Any,
    labels: Any,
    np: Any,
) -> None:
    if isinstance(logits, tuple):
        logits = logits[0]
    probabilities = sigmoid(logits, np)
    predictions = (probabilities >= 0.5).astype(int)
    labels = labels.astype(int)

    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    texts = val_frame[args.text_column].astype(str).tolist()
    for row_index in range(labels.shape[0]):
        for label_index, label_name in enumerate(LABEL_COLUMNS):
            probability = float(probabilities[row_index][label_index])
            true_value = int(labels[row_index][label_index])
            predicted_value = int(predictions[row_index][label_index])
            row = {
                "validation_index": row_index,
                "label": label_name,
                "probability": round(probability, 6),
                "true_value": true_value,
                "predicted_value": predicted_value,
                "comment_text": texts[row_index],
            }
            if predicted_value == 1 and true_value == 0:
                false_positives.append({**row, "confidence": round(probability, 6)})
            elif predicted_value == 0 and true_value == 1:
                false_negatives.append({**row, "confidence": round(1.0 - probability, 6)})

    false_positives.sort(key=lambda row: float(row["confidence"]), reverse=True)
    false_negatives.sort(key=lambda row: float(row["confidence"]), reverse=True)
    limit = max(0, int(args.error_example_limit))
    save_error_csv(setting_dir / "false_positive_examples.csv", false_positives[:limit])
    save_error_csv(setting_dir / "false_negative_examples.csv", false_negatives[:limit])


def save_threshold_tuning(
    setting_dir: Path,
    logits: Any,
    labels: Any,
    np: Any,
    f1_score: Any,
) -> None:
    """Tune one validation threshold per label and write comparison artifacts."""

    if isinstance(logits, tuple):
        logits = logits[0]
    probabilities = sigmoid(logits, np)
    labels = labels.astype(int)
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)

    per_label_rows: list[dict[str, Any]] = []
    thresholds = []
    for label_index, label_name in enumerate(LABEL_COLUMNS):
        y_true = labels[:, label_index]
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in grid:
            y_pred = (probabilities[:, label_index] >= threshold).astype(int)
            score = float(f1_score(y_true, y_pred, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
        base_pred = (probabilities[:, label_index] >= 0.5).astype(int)
        per_label_rows.append(
            {
                "label": label_name,
                "base_threshold": 0.5,
                "base_f1": float(f1_score(y_true, base_pred, zero_division=0)),
                "tuned_threshold": best_threshold,
                "tuned_f1": best_f1,
                "positive_examples": int(y_true.sum()),
            }
        )

    base_predictions = (probabilities >= 0.5).astype(int)
    tuned_predictions = np.zeros_like(base_predictions)
    for label_index, threshold in enumerate(thresholds):
        tuned_predictions[:, label_index] = (
            probabilities[:, label_index] >= threshold
        ).astype(int)

    summary = {
        "base_threshold": 0.5,
        "tuned_thresholds": {
            row["label"]: row["tuned_threshold"] for row in per_label_rows
        },
        "base_macro_f1": float(
            f1_score(labels, base_predictions, average="macro", zero_division=0)
        ),
        "base_micro_f1": float(
            f1_score(labels, base_predictions, average="micro", zero_division=0)
        ),
        "tuned_macro_f1": float(
            f1_score(labels, tuned_predictions, average="macro", zero_division=0)
        ),
        "tuned_micro_f1": float(
            f1_score(labels, tuned_predictions, average="micro", zero_division=0)
        ),
        "threshold_grid": [float(value) for value in grid],
    }

    setting_dir.mkdir(parents=True, exist_ok=True)
    with (setting_dir / "threshold_tuning_per_label.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_label_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_label_rows)
    save_json(setting_dir / "threshold_tuning_summary.json", summary)


def run_setting(
    setting_name: str,
    max_length: int,
    args: argparse.Namespace,
    train_frame: Any,
    val_frame: Any,
    deps: dict[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    np = deps["np"]
    torch = deps["torch"]
    AutoTokenizer = deps["AutoTokenizer"]
    AutoModelForSequenceClassification = deps["AutoModelForSequenceClassification"]
    DataCollatorWithPadding = deps["DataCollatorWithPadding"]
    Trainer = deps["Trainer"]
    set_seed = deps["set_seed"]

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=len(LABEL_COLUMNS),
        problem_type="multi_label_classification",
        local_files_only=args.local_files_only,
    )

    train_dataset = make_dataset(train_frame, tokenizer, args, max_length, np, torch)
    val_dataset = make_dataset(val_frame, tokenizer, args, max_length, np, torch)
    output_dir = artifact_root / setting_name / "checkpoints"
    training_args = make_training_arguments(deps["TrainingArguments"], args, output_dir)
    timer, epoch_seconds = EpochTimerFactory(deps["TrainerCallback"]).make()
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": build_metric_fn(np, deps["f1_score"], deps["roc_auc_score"]),
        "callbacks": [timer],
    }
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

    started_at = time.perf_counter()
    train_output = trainer.train()
    total_train_seconds = time.perf_counter() - started_at
    validation = trainer.evaluate()
    average_epoch_seconds = float(sum(epoch_seconds) / len(epoch_seconds)) if epoch_seconds else total_train_seconds

    result = {
        "setting": setting_name,
        "checkpoint": args.checkpoint,
        "tokenizer": args.checkpoint,
        "max_length": int(max_length),
        "train_examples": int(len(train_dataset)),
        "validation_examples": int(len(val_dataset)),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "epochs": float(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "mixed_precision": args.mixed_precision,
        "trainable_parameters": count_trainable_parameters(model),
        "train_seconds_total": float(total_train_seconds),
        "epoch_seconds": [float(value) for value in epoch_seconds],
        "epoch_seconds_average": average_epoch_seconds,
        "epoch_time": format_hms(average_epoch_seconds),
        "train_loss": float(getattr(train_output, "training_loss", float("nan"))),
        "validation": {
            key.removeprefix("eval_"): float(value)
            for key, value in validation.items()
            if isinstance(value, numbers.Real)
        },
    }
    result["validation_macro_f1"] = float(result["validation"].get("macro_f1", float("nan")))
    result["validation_micro_f1"] = float(result["validation"].get("micro_f1", float("nan")))
    result["validation_macro_roc_auc"] = float(result["validation"].get("macro_roc_auc", float("nan")))

    if args.save_artifacts:
        result["artifacts"] = {
            "false_positive_examples": str(artifact_root / setting_name / "false_positive_examples.csv"),
            "false_negative_examples": str(artifact_root / setting_name / "false_negative_examples.csv"),
        }
        prediction_output = trainer.predict(val_dataset)
        save_error_examples(
            artifact_root / setting_name,
            val_frame,
            args,
            prediction_output.predictions,
            prediction_output.label_ids,
            np,
        )
        save_threshold_tuning(
            artifact_root / setting_name,
            prediction_output.predictions,
            prediction_output.label_ids,
            np,
            deps["f1_score"],
        )

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_comparison_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "setting",
        "max_length",
        "validation_macro_f1",
        "validation_micro_f1",
        "validation_macro_roc_auc",
        "epoch_time",
        "epoch_seconds_average",
        "train_seconds_total",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()
    apply_quick_settings(args)
    deps = load_dependencies(args)
    if deps is None:
        return 0
    if args.check_deps:
        print("Dependency check passed.")
        return 0

    np = deps["np"]
    pd = deps["pd"]

    if args.synthetic_data:
        frame = make_synthetic_frame(args, pd)
        dataset_source = "synthetic"
    else:
        frame = load_kaggle_frame(args, pd)
        dataset_source = str(Path(args.data_dir) / args.train_csv)

    train_frame, val_frame = split_frame(args, frame, np)
    artifact_root = Path(args.artifact_dir)
    results = []
    for index, max_length in enumerate(args.max_lengths):
        setting_name = chr(ord("A") + index)
        results.append(run_setting(setting_name, max_length, args, train_frame, val_frame, deps, artifact_root))

    summary = {
        "command": " ".join(sys.argv),
        "dataset_source": dataset_source,
        "label_columns": LABEL_COLUMNS,
        "seed": int(args.seed),
        "validation_fraction": float(args.validation_fraction),
        "versions": package_versions(deps),
        "results": results,
    }

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.save_artifacts:
        save_json(artifact_root / "run_summary.json", summary)
        save_json(artifact_root / "versions.json", summary["versions"])
        (artifact_root / "command.txt").write_text(summary["command"] + "\n", encoding="utf-8")
        save_comparison_csv(artifact_root / "comparison_table.csv", results)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
