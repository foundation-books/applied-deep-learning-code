#!/usr/bin/env python3
"""Train a non-Transformer TF-IDF baseline for toxic-comment classification."""

from __future__ import annotations

import argparse
import csv
import json
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
    parser.add_argument("--data-dir", default="data", help="Directory containing train.csv.")
    parser.add_argument("--train-csv", default="train.csv")
    parser.add_argument("--text-column", default="comment_text")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--artifact-dir", default="runs/toxic-tfidf-baseline")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--allow-missing-deps", action="store_true")
    return parser.parse_args()


def load_dependencies(args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.metrics import f1_score, roc_auc_score
        from sklearn.multiclass import OneVsRestClassifier
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
        "sklearn": sklearn,
        "TfidfVectorizer": TfidfVectorizer,
        "SGDClassifier": SGDClassifier,
        "OneVsRestClassifier": OneVsRestClassifier,
        "f1_score": f1_score,
        "roc_auc_score": roc_auc_score,
    }


def load_frame(args: argparse.Namespace, pd: Any) -> Any:
    path = Path(args.data_dir) / args.train_csv
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    frame = pd.read_csv(path)
    required = {args.text_column, *LABEL_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame[[args.text_column, *LABEL_COLUMNS]].copy()


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


def evaluate(labels: Any, probabilities: Any, f1_score: Any, roc_auc_score: Any) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "per_label_f1": {
            label: float(score)
            for label, score in zip(
                LABEL_COLUMNS,
                f1_score(labels, predictions, average=None, zero_division=0),
            )
        },
    }
    try:
        metrics["macro_roc_auc"] = float(roc_auc_score(labels, probabilities, average="macro"))
    except ValueError:
        metrics["macro_roc_auc"] = float("nan")
    return metrics


def package_versions(deps: dict[str, Any]) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": str(getattr(deps["np"], "__version__", "unknown")),
        "pandas": str(getattr(deps["pd"], "__version__", "unknown")),
        "sklearn": str(getattr(deps["sklearn"], "__version__", "unknown")),
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_comparison_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "method",
        "train_examples",
        "validation_examples",
        "max_features",
        "ngram_max",
        "validation_macro_f1",
        "validation_micro_f1",
        "validation_macro_roc_auc",
        "train_seconds_total",
    ]
    row = {field: result.get(field, "") for field in fields}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    deps = load_dependencies(args)
    if deps is None:
        return 0
    if args.check_deps:
        print("Dependency check passed.")
        return 0

    np = deps["np"]
    pd = deps["pd"]
    frame = load_frame(args, pd)
    train_frame, val_frame = split_frame(args, frame, np)
    vectorizer = deps["TfidfVectorizer"](
        lowercase=True,
        strip_accents="unicode",
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=2,
        dtype=np.float32,
    )
    classifier = deps["OneVsRestClassifier"](
        deps["SGDClassifier"](
            loss="log_loss",
            alpha=args.alpha,
            max_iter=args.max_iter,
            tol=1e-3,
            random_state=args.seed,
            class_weight="balanced",
        )
    )

    y_train = train_frame[LABEL_COLUMNS].to_numpy(dtype=np.int32)
    y_val = val_frame[LABEL_COLUMNS].to_numpy(dtype=np.int32)
    started_at = time.perf_counter()
    x_train = vectorizer.fit_transform(train_frame[args.text_column].astype(str))
    classifier.fit(x_train, y_train)
    train_seconds = time.perf_counter() - started_at
    x_val = vectorizer.transform(val_frame[args.text_column].astype(str))
    probabilities = classifier.predict_proba(x_val)
    metrics = evaluate(y_val, probabilities, deps["f1_score"], deps["roc_auc_score"])

    result = {
        "run": "baseline",
        "method": "tfidf_sgd_logistic",
        "train_examples": int(len(train_frame)),
        "validation_examples": int(len(val_frame)),
        "max_features": int(args.max_features),
        "ngram_max": int(args.ngram_max),
        "alpha": float(args.alpha),
        "max_iter": int(args.max_iter),
        "train_seconds_total": float(train_seconds),
        "validation": metrics,
        "validation_macro_f1": float(metrics["macro_f1"]),
        "validation_micro_f1": float(metrics["micro_f1"]),
        "validation_macro_roc_auc": float(metrics["macro_roc_auc"]),
    }
    summary = {
        "command": " ".join(sys.argv),
        "dataset_source": str(Path(args.data_dir) / args.train_csv),
        "seed": int(args.seed),
        "validation_fraction": float(args.validation_fraction),
        "versions": package_versions(deps),
        "result": result,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.save_artifacts:
        artifact_root = Path(args.artifact_dir)
        save_json(artifact_root / "baseline_summary.json", summary)
        save_comparison_csv(artifact_root / "baseline_table.csv", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
