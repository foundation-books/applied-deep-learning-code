#!/usr/bin/env python3
"""Audit Jigsaw toxic-comment labels and truncation rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
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
    parser.add_argument("--max-lengths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--checkpoint", default="distilbert-base-uncased")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick audits.")
    parser.add_argument("--artifact-dir", default="runs/toxic-data-audit")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument(
        "--allow-missing-deps",
        action="store_true",
        help="Return success when dependency imports fail; useful for repository checks.",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Only compute character/word-length summaries; do not load a Transformer tokenizer.",
    )
    return parser.parse_args()


def load_dependencies(args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        import numpy as np
        import pandas as pd
        transformers = None
        if not args.skip_tokenizer:
            import transformers
    except Exception as exc:  # pragma: no cover - depends on local environment
        message = f"Missing or unusable dependency: {exc}"
        if args.allow_missing_deps:
            print(message)
            print("Dependency check skipped because --allow-missing-deps was supplied.")
            return None
        raise RuntimeError(message) from exc

    return {"np": np, "pd": pd, "transformers": transformers}


def load_frame(args: argparse.Namespace, pd: Any) -> Any:
    path = Path(args.data_dir) / args.train_csv
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    frame = pd.read_csv(path)
    required = {args.text_column, *LABEL_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame[[args.text_column, *LABEL_COLUMNS]].copy()
    if args.limit is not None:
        frame = frame.head(args.limit).copy()
    return frame


def split_indices(count: int, validation_fraction: float, seed: int, np: Any) -> tuple[Any, Any]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    indices = np.arange(count)
    rng.shuffle(indices)
    val_count = max(1, int(round(count * validation_fraction)))
    return indices[val_count:], indices[:val_count]


def label_summary(frame: Any) -> dict[str, Any]:
    rows = len(frame)
    positives = frame[LABEL_COLUMNS].sum(axis=0)
    any_toxic = frame[LABEL_COLUMNS].sum(axis=1).gt(0).sum()
    per_label = {}
    for label in LABEL_COLUMNS:
        count = int(positives[label])
        per_label[label] = {
            "positive": count,
            "negative": int(rows - count),
            "positive_rate": float(count / rows) if rows else 0.0,
        }
    return {
        "examples": int(rows),
        "any_label_positive": int(any_toxic),
        "any_label_positive_rate": float(any_toxic / rows) if rows else 0.0,
        "per_label": per_label,
    }


def length_summary(frame: Any, text_column: str, np: Any) -> dict[str, Any]:
    texts = frame[text_column].astype(str)
    char_lengths = texts.str.len().to_numpy()
    word_lengths = texts.str.split().map(len).to_numpy()

    def describe(values: Any) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        }

    return {"characters": describe(char_lengths), "whitespace_words": describe(word_lengths)}


def tokenizer_summary(frame: Any, args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_tokenizer:
        return {"skipped": True}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    texts = frame[args.text_column].astype(str).tolist()
    token_lengths = [
        len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])
        for text in texts
    ]
    total = len(token_lengths)
    return {
        "checkpoint": args.checkpoint,
        "token_lengths": {
            "mean": float(sum(token_lengths) / total) if total else 0.0,
            "max": int(max(token_lengths)) if total else 0,
        },
        "max_length_truncation": {
            str(max_length): {
                "truncated_examples": int(sum(length > max_length for length in token_lengths)),
                "truncated_rate": float(sum(length > max_length for length in token_lengths) / total) if total else 0.0,
            }
            for max_length in args.max_lengths
        },
    }


def package_versions(deps: dict[str, Any]) -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name, module in deps.items():
        versions[name] = str(getattr(module, "__version__", "not imported"))
    return versions


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    train_indices, val_indices = split_indices(len(frame), args.validation_fraction, args.seed, np)
    train_frame = frame.iloc[train_indices].reset_index(drop=True)
    val_frame = frame.iloc[val_indices].reset_index(drop=True)

    summary = {
        "command": " ".join(sys.argv),
        "dataset_source": str(Path(args.data_dir) / args.train_csv),
        "seed": int(args.seed),
        "validation_fraction": float(args.validation_fraction),
        "versions": package_versions(deps),
        "all": {
            "labels": label_summary(frame),
            "lengths": length_summary(frame, args.text_column, np),
            "tokenizer": tokenizer_summary(frame, args),
        },
        "train": {"labels": label_summary(train_frame)},
        "validation": {"labels": label_summary(val_frame)},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.save_artifacts:
        save_json(Path(args.artifact_dir) / "data_audit_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
