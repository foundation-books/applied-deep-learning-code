#!/usr/bin/env python3
"""Run the Transformer distillation reference experiment for the chapter.

The default run uses AG News, a task-finetuned BERT teacher, and a smaller BERT
student. It produces compact CSV and JSON artifacts that can be copied into the
chapter figures directory and cited from the case-study prose.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import random
import shlex
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
CSV_LINETERMINATOR = "\n"
ARTIFACT_PREFIX = "nntrev-reference"

INSTALL_HELP = """Install the experiment dependencies first:
  python3 -m pip install 'torch>=2.2' 'transformers>=4.40,<5' 'datasets>=2.19' numpy matplotlib plotnine
"""


def load_plot_style() -> Any:
    import importlib.util

    for parent in Path(__file__).resolve().parents:
        for candidate in (
            parent / "adl_plot_style.py",
            parent / "book" / "_maintenance" / "plot_style.py",
        ):
            if not candidate.exists():
                continue
            spec = importlib.util.spec_from_file_location("adl_plot_style", candidate)
            if spec is None or spec.loader is None:
                break
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Could not locate adl_plot_style.py")


PLOT_STYLE = load_plot_style()


@dataclass
class EncodedSplit:
    texts: list[str]
    labels: list[int]
    encodings: dict[str, Any]


@dataclass
class ModelResult:
    key: str
    display_name: str
    validation_accuracy: float
    validation_loss: float
    test_accuracy: float
    test_loss: float
    parameter_count: int
    trainable_parameter_count: int
    model_size_mb: float
    training_time_seconds: float
    peak_memory_allocated_mb: float | None
    peak_memory_reserved_mb: float | None
    notes: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def alpha_value(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["ag_news", "synthetic"], default="ag_news"
    )
    parser.add_argument(
        "--teacher-checkpoint", default="textattack/bert-base-uncased-ag-news"
    )
    parser.add_argument("--student-checkpoint", default="prajjwal1/bert-tiny")
    parser.add_argument(
        "--tokenizer-checkpoint",
        default=None,
        help="Tokenizer checkpoint. Defaults to the student checkpoint.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/nntrev-reference-ag-news")
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-length", type=positive_int, default=96)
    parser.add_argument("--train-examples", type=positive_int, default=4000)
    parser.add_argument("--validation-examples", type=positive_int, default=1000)
    parser.add_argument("--test-examples", type=positive_int, default=1000)
    parser.add_argument("--teacher-epochs", type=nonnegative_int, default=0)
    parser.add_argument("--student-epochs", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--eval-batch-size", type=positive_int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=1)
    parser.add_argument("--teacher-learning-rate", type=positive_float, default=2e-5)
    parser.add_argument("--student-learning-rate", type=positive_float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--temperature", type=positive_float, default=2.0)
    parser.add_argument(
        "--alpha",
        type=alpha_value,
        default=0.5,
        help="Weight on hard-label cross-entropy in the distillation loss.",
    )
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument(
        "--mixed-precision",
        choices=["none", "fp16", "bf16"],
        default="none",
        help="Use CUDA autocast when the device supports it.",
    )
    parser.add_argument("--latency-warmup", type=nonnegative_int, default=8)
    parser.add_argument("--latency-repetitions", type=positive_int, default=40)
    parser.add_argument(
        "--latency-batch-sizes", type=positive_int, nargs="+", default=[1, 32]
    )
    parser.add_argument("--calibration-bins", type=positive_int, default=10)
    parser.add_argument("--error-examples", type=nonnegative_int, default=30)
    parser.add_argument(
        "--include-error-text",
        action="store_true",
        help=(
            "Include raw test-set text in the error-example CSV. Leave disabled "
            "for public artifacts unless the dataset terms permit redistribution."
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use synthetic data and tiny random checkpoints for a fast pipeline check.",
    )
    parser.add_argument(
        "--check-deps", action="store_true", help="Import dependencies and exit."
    )
    parser.add_argument(
        "--allow-missing-deps",
        action="store_true",
        help="Return success when dependency imports fail; useful for repository checks.",
    )
    return parser.parse_args(argv)


def apply_quick_settings(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.dataset = "synthetic"
    args.teacher_checkpoint = "hf-internal-testing/tiny-random-bert"
    args.student_checkpoint = "hf-internal-testing/tiny-random-bert"
    args.tokenizer_checkpoint = "hf-internal-testing/tiny-random-bert"
    args.train_examples = min(args.train_examples, 64)
    args.validation_examples = min(args.validation_examples, 32)
    args.test_examples = min(args.test_examples, 32)
    args.teacher_epochs = min(args.teacher_epochs, 1)
    args.student_epochs = 1
    args.batch_size = min(args.batch_size, 8)
    args.eval_batch_size = min(args.eval_batch_size, 16)
    args.max_length = min(args.max_length, 32)
    args.latency_warmup = min(args.latency_warmup, 2)
    args.latency_repetitions = min(args.latency_repetitions, 4)


def load_plotnine_dependencies() -> dict[str, Any] | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import pandas as pd
        import plotnine as p9
        from plotnine import (
            aes,
            coord_fixed,
            element_blank,
            element_line,
            element_rect,
            element_text,
            facet_wrap,
            geom_col,
            geom_line,
            geom_point,
            geom_text,
            geom_tile,
            ggplot,
            labs,
            position_dodge,
            scale_color_identity,
            scale_color_manual,
            scale_fill_gradient,
            scale_fill_manual,
            scale_size_continuous,
            scale_x_discrete,
            scale_y_continuous,
            scale_y_discrete,
            theme,
            theme_minimal,
        )
    except Exception:  # pragma: no cover - optional plotting
        return None

    return {
        "pd": pd,
        "aes": aes,
        "coord_fixed": coord_fixed,
        "element_blank": element_blank,
        "element_line": element_line,
        "element_rect": element_rect,
        "element_text": element_text,
        "facet_wrap": facet_wrap,
        "geom_col": geom_col,
        "geom_line": geom_line,
        "geom_point": geom_point,
        "geom_text": geom_text,
        "geom_tile": geom_tile,
        "ggplot": ggplot,
        "labs": labs,
        "p9": p9,
        "position_dodge": position_dodge,
        "scale_color_identity": scale_color_identity,
        "scale_color_manual": scale_color_manual,
        "scale_fill_gradient": scale_fill_gradient,
        "scale_fill_manual": scale_fill_manual,
        "scale_size_continuous": scale_size_continuous,
        "scale_x_discrete": scale_x_discrete,
        "scale_y_continuous": scale_y_continuous,
        "scale_y_discrete": scale_y_discrete,
        "style": PLOT_STYLE,
        "theme": theme,
        "theme_minimal": theme_minimal,
    }


def load_dependencies(args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        from datasets import load_dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
            set_seed,
        )
    except Exception as exc:  # pragma: no cover - depends on local environment
        message = f"Missing or unusable dependency: {exc}"
        if args.allow_missing_deps:
            print(message)
            print("Dependency check skipped because --allow-missing-deps was supplied.")
            return None
        raise RuntimeError(message + "\n\n" + INSTALL_HELP) from exc

    return {
        "np": np,
        "torch": torch,
        "F": F,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "load_dataset": load_dataset,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
        "set_seed": set_seed,
        "plotnine": load_plotnine_dependencies(),
    }


def seed_everything(args: argparse.Namespace, deps: dict[str, Any]) -> None:
    random.seed(args.seed)
    deps["np"].random.seed(args.seed)
    deps["torch"].manual_seed(args.seed)
    if deps["torch"].cuda.is_available():
        deps["torch"].cuda.manual_seed_all(args.seed)
    deps["set_seed"](args.seed)


def make_synthetic_rows(count: int, seed: int) -> tuple[list[str], list[int]]:
    templates = [
        ("world leaders met for trade and security talks", 0),
        ("the national team won the final after extra time", 1),
        ("stocks rose after the company reported higher earnings", 2),
        ("a new satellite sensor improved climate measurements", 3),
        ("diplomats discussed a ceasefire at the summit", 0),
        ("the coach changed the lineup before the tournament", 1),
        ("the central bank signaled a change in interest rates", 2),
        ("researchers released a faster chip for mobile devices", 3),
    ]
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[int] = []
    for index in range(count):
        text, label = templates[index % len(templates)]
        suffix = rng.choice(["today", "this week", "in a short report", "after review"])
        texts.append(f"{text} {suffix} example {index}")
        labels.append(label)
    return texts, labels


def select_split(split: Any, start: int, count: int) -> Any:
    end = min(len(split), start + count)
    if end <= start:
        raise ValueError("requested split has no examples after selection")
    return split.select(range(start, end))


def load_text_splits(
    args: argparse.Namespace, deps: dict[str, Any]
) -> dict[str, tuple[list[str], list[int]]]:
    if args.dataset == "synthetic":
        train = make_synthetic_rows(args.train_examples, args.seed)
        validation = make_synthetic_rows(args.validation_examples, args.seed + 1)
        test = make_synthetic_rows(args.test_examples, args.seed + 2)
        return {"train": train, "validation": validation, "test": test}

    raw = deps["load_dataset"](
        "ag_news", cache_dir=str(args.cache_dir) if args.cache_dir else None
    )
    shuffled_train = raw["train"].shuffle(seed=args.seed)
    shuffled_test = raw["test"].shuffle(seed=args.seed)
    validation_split = select_split(shuffled_train, 0, args.validation_examples)
    train_split = select_split(
        shuffled_train, args.validation_examples, args.train_examples
    )
    test_split = select_split(shuffled_test, 0, args.test_examples)
    return {
        "train": (list(train_split["text"]), [int(x) for x in train_split["label"]]),
        "validation": (
            list(validation_split["text"]),
            [int(x) for x in validation_split["label"]],
        ),
        "test": (list(test_split["text"]), [int(x) for x in test_split["label"]]),
    }


def encode_split(
    tokenizer: Any, texts: list[str], labels: list[int], args: argparse.Namespace
) -> EncodedSplit:
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    return EncodedSplit(texts=texts, labels=labels, encodings=dict(encodings))


def make_dataset_class(deps: dict[str, Any]) -> Any:
    torch = deps["torch"]
    Dataset = deps["Dataset"]

    class TextClassificationDataset(Dataset):
        def __init__(self, split: EncodedSplit) -> None:
            self.split = split

        def __len__(self) -> int:
            return len(self.split.labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = {key: value[index] for key, value in self.split.encodings.items()}
            item["labels"] = torch.tensor(self.split.labels[index], dtype=torch.long)
            item["index"] = torch.tensor(index, dtype=torch.long)
            return item

    return TextClassificationDataset


def make_loaders(
    encoded: dict[str, EncodedSplit],
    args: argparse.Namespace,
    deps: dict[str, Any],
) -> dict[str, Any]:
    DataLoader = deps["DataLoader"]
    dataset_cls = make_dataset_class(deps)
    generator = deps["torch"].Generator()
    generator.manual_seed(args.seed)
    return {
        "train": DataLoader(
            dataset_cls(encoded["train"]),
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
        ),
        "validation": DataLoader(
            dataset_cls(encoded["validation"]),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
        "test": DataLoader(
            dataset_cls(encoded["test"]),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
        "train_eval": DataLoader(
            dataset_cls(encoded["train"]),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
    }


def load_sequence_classifier(
    checkpoint: str,
    args: argparse.Namespace,
    deps: dict[str, Any],
    *,
    num_labels: int = 4,
) -> Any:
    model_cls = deps["AutoModelForSequenceClassification"]
    kwargs = {
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "local_files_only": args.local_files_only,
    }
    return model_cls.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        **kwargs,
    )


def batch_to_device(
    batch: dict[str, Any], device: Any
) -> tuple[dict[str, Any], Any, Any]:
    labels = batch["labels"].to(device)
    indices = batch["index"].to(device)
    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key not in {"labels", "index"}
    }
    return inputs, labels, indices


def autocast_context(
    args: argparse.Namespace, deps: dict[str, Any], device: Any
) -> Any:
    torch = deps["torch"]
    if device.type != "cuda" or args.mixed_precision == "none":
        return nullcontext()
    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def reset_peak_memory(deps: dict[str, Any], device: Any) -> None:
    torch = deps["torch"]
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass


def read_peak_memory(
    deps: dict[str, Any], device: Any
) -> tuple[float | None, float | None]:
    torch = deps["torch"]
    if device.type != "cuda":
        return None, None
    try:
        allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
        reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
    except Exception:
        return None, None
    return allocated, reserved


def build_optimizer_and_scheduler(
    model: Any,
    loader: Any,
    epochs: int,
    learning_rate: float,
    args: argparse.Namespace,
    deps: dict[str, Any],
) -> tuple[Any, Any]:
    torch = deps["torch"]
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    grouped = [
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if param.requires_grad and not any(nd in name for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if param.requires_grad and any(nd in name for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=learning_rate)
    total_steps = max(
        1, math.ceil(len(loader) / args.gradient_accumulation_steps) * max(1, epochs)
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = deps["get_linear_schedule_with_warmup"](
        optimizer, warmup_steps, total_steps
    )
    return optimizer, scheduler


def evaluate_model(
    model: Any, loader: Any, args: argparse.Namespace, deps: dict[str, Any], device: Any
) -> dict[str, Any]:
    torch = deps["torch"]
    F = deps["F"]
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_logits: list[Any] = []
    all_labels: list[Any] = []
    with torch.no_grad():
        for batch in loader:
            inputs, labels, _ = batch_to_device(batch, device)
            with autocast_context(args, deps, device):
                logits = model(**inputs).logits
                loss = F.cross_entropy(logits.float(), labels)
            count = labels.numel()
            total_loss += float(loss.item()) * count
            total_examples += count
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(labels.detach().cpu())
    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    predictions = logits_tensor.argmax(dim=1)
    accuracy = float((predictions == labels_tensor).float().mean().item())
    probabilities = torch.softmax(logits_tensor, dim=1)
    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": accuracy,
        "logits": logits_tensor,
        "labels": labels_tensor,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def collect_logits(
    model: Any, loader: Any, args: argparse.Namespace, deps: dict[str, Any], device: Any
) -> Any:
    return evaluate_model(model, loader, args, deps, device)["logits"]


def train_model(
    model: Any,
    loaders: dict[str, Any],
    args: argparse.Namespace,
    deps: dict[str, Any],
    device: Any,
    *,
    epochs: int,
    learning_rate: float,
    model_key: str,
    teacher_logits: Any | None = None,
) -> tuple[list[dict[str, Any]], float, float | None, float | None]:
    torch = deps["torch"]
    F = deps["F"]
    model.to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, loaders["train"], epochs, learning_rate, args, deps
    )
    use_scaler = device.type == "cuda" and args.mixed_precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    history: list[dict[str, Any]] = []
    reset_peak_memory(deps, device)
    start_time = time.perf_counter()
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_examples = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loaders["train"], start=1):
            inputs, labels, indices = batch_to_device(batch, device)
            with autocast_context(args, deps, device):
                logits = model(**inputs).logits
                hard_loss = F.cross_entropy(logits.float(), labels)
                if teacher_logits is None:
                    loss = hard_loss
                else:
                    batch_teacher = teacher_logits[indices.detach().cpu()].to(device)
                    temperature = args.temperature
                    soft_loss = F.kl_div(
                        F.log_softmax(logits.float() / temperature, dim=-1),
                        F.softmax(batch_teacher.float() / temperature, dim=-1),
                        reduction="batchmean",
                    ) * (temperature * temperature)
                    loss = args.alpha * hard_loss + (1.0 - args.alpha) * soft_loss
                loss = loss / args.gradient_accumulation_steps
            count = labels.numel()
            running_loss += (
                float(loss.item()) * args.gradient_accumulation_steps * count
            )
            running_examples += count
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(
                loaders["train"]
            ):
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        validation = evaluate_model(model, loaders["validation"], args, deps, device)
        history.append(
            {
                "model": model_key,
                "epoch": epoch,
                "train_loss": running_loss / max(1, running_examples),
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "global_step": global_step,
            }
        )
    elapsed = time.perf_counter() - start_time
    peak_allocated, peak_reserved = read_peak_memory(deps, device)
    return history, elapsed, peak_allocated, peak_reserved


def parameter_count(model: Any) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    return int(total), int(trainable)


def estimate_model_size_mb(
    model: Any, output_dir: Path, key: str, deps: dict[str, Any], save_models: bool
) -> float:
    torch = deps["torch"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_models:
        path = output_dir / f"{ARTIFACT_PREFIX}-{key}-state-dict.pt"
        torch.save(model.state_dict(), path)
        return path.stat().st_size / (1024**2)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        torch.save(model.state_dict(), tmp_path)
        return tmp_path.stat().st_size / (1024**2)
    finally:
        tmp_path.unlink(missing_ok=True)


def measure_latency(
    model: Any,
    encoded_test: EncodedSplit,
    args: argparse.Namespace,
    deps: dict[str, Any],
    device: Any,
    *,
    model_key: str,
    display_name: str,
) -> list[dict[str, Any]]:
    torch = deps["torch"]
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for requested_batch_size in args.latency_batch_sizes:
            batch_size = min(requested_batch_size, len(encoded_test.labels))
            if batch_size <= 0:
                continue
            inputs = {
                key: value[:batch_size].to(device)
                for key, value in encoded_test.encodings.items()
            }
            for _ in range(args.latency_warmup):
                with autocast_context(args, deps, device):
                    _ = model(**inputs).logits
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            per_example_ms: list[float] = []
            for _ in range(args.latency_repetitions):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                with autocast_context(args, deps, device):
                    _ = model(**inputs).logits
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                per_example_ms.append((elapsed * 1000.0) / batch_size)
            sorted_ms = sorted(per_example_ms)
            mean_ms = sum(per_example_ms) / len(per_example_ms)
            p50 = sorted_ms[len(sorted_ms) // 2]
            p95 = sorted_ms[
                min(len(sorted_ms) - 1, int(math.ceil(0.95 * len(sorted_ms))) - 1)
            ]
            rows.append(
                {
                    "model": model_key,
                    "display_name": display_name,
                    "batch_size": batch_size,
                    "requested_batch_size": requested_batch_size,
                    "mean_latency_ms_per_example": mean_ms,
                    "p50_latency_ms_per_example": p50,
                    "p95_latency_ms_per_example": p95,
                    "throughput_examples_per_second": 1000.0 / mean_ms
                    if mean_ms > 0
                    else 0.0,
                    "warmup_repetitions": args.latency_warmup,
                    "timed_repetitions": args.latency_repetitions,
                    "device": str(device),
                    "precision": args.mixed_precision,
                    "max_length": args.max_length,
                }
            )
    return rows


def calibration_summary(
    probabilities: Any,
    labels: Any,
    bins: int,
    deps: dict[str, Any],
) -> dict[str, float]:
    np = deps["np"]
    probs = probabilities.numpy()
    y = labels.numpy()
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y).astype(float)
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        if bin_index == bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not mask.any():
            continue
        bin_accuracy = correct[mask].mean()
        bin_confidence = confidences[mask].mean()
        ece += (mask.mean()) * abs(float(bin_accuracy) - float(bin_confidence))
    clipped = np.clip(probs, 1e-12, 1.0)
    nll = -np.log(clipped[np.arange(len(y)), y]).mean()
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y)), y] = 1.0
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return {
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidences.mean()),
        "expected_calibration_error": float(ece),
        "negative_log_likelihood": float(nll),
        "brier_score": float(brier),
    }


def confusion_rows(
    predictions: Any,
    labels: Any,
    model_key: str,
    display_name: str,
    deps: dict[str, Any],
) -> list[dict[str, Any]]:
    np = deps["np"]
    matrix = np.zeros((len(LABEL_NAMES), len(LABEL_NAMES)), dtype=int)
    for true_label, predicted_label in zip(labels.numpy(), predictions.numpy()):
        matrix[int(true_label), int(predicted_label)] += 1
    rows: list[dict[str, Any]] = []
    for true_index, true_name in enumerate(LABEL_NAMES):
        for pred_index, pred_name in enumerate(LABEL_NAMES):
            rows.append(
                {
                    "model": model_key,
                    "display_name": display_name,
                    "true_label": true_name,
                    "predicted_label": pred_name,
                    "count": int(matrix[true_index, pred_index]),
                }
            )
    return rows


def make_error_examples(
    encoded_test: EncodedSplit,
    teacher_eval: dict[str, Any],
    hard_eval: dict[str, Any],
    distilled_eval: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.error_examples <= 0:
        return rows
    labels = teacher_eval["labels"].numpy()
    teacher_pred = teacher_eval["predictions"].numpy()
    hard_pred = hard_eval["predictions"].numpy()
    distilled_pred = distilled_eval["predictions"].numpy()
    teacher_conf = teacher_eval["probabilities"].max(dim=1).values.numpy()
    hard_conf = hard_eval["probabilities"].max(dim=1).values.numpy()
    distilled_conf = distilled_eval["probabilities"].max(dim=1).values.numpy()
    preferred_categories = [
        "teacher_correct_distilled_wrong",
        "distilled_correct_hard_wrong",
        "hard_correct_distilled_wrong",
        "teacher_wrong_distilled_correct",
        "all_wrong",
    ]
    counts = {category: 0 for category in preferred_categories}

    def category_for(i: int) -> str | None:
        t_ok = teacher_pred[i] == labels[i]
        h_ok = hard_pred[i] == labels[i]
        d_ok = distilled_pred[i] == labels[i]
        if t_ok and not d_ok:
            return "teacher_correct_distilled_wrong"
        if d_ok and not h_ok:
            return "distilled_correct_hard_wrong"
        if h_ok and not d_ok:
            return "hard_correct_distilled_wrong"
        if (not t_ok) and d_ok:
            return "teacher_wrong_distilled_correct"
        if not t_ok and not h_ok and not d_ok:
            return "all_wrong"
        return None

    per_category_target = max(
        1, math.ceil(args.error_examples / len(preferred_categories))
    )

    def example_text(text: str) -> str:
        if args.include_error_text:
            return " ".join(text.split())[:300]
        return "[redacted: original dataset text is not redistributed in public artifacts]"

    for index, text in enumerate(encoded_test.texts):
        category = category_for(index)
        if category is None or counts[category] >= per_category_target:
            continue
        counts[category] += 1
        rows.append(
            {
                "example_index": index,
                "category": category,
                "text": example_text(text),
                "true_label": LABEL_NAMES[int(labels[index])],
                "teacher_prediction": LABEL_NAMES[int(teacher_pred[index])],
                "hard_student_prediction": LABEL_NAMES[int(hard_pred[index])],
                "distilled_student_prediction": LABEL_NAMES[int(distilled_pred[index])],
                "teacher_confidence": float(teacher_conf[index]),
                "hard_student_confidence": float(hard_conf[index]),
                "distilled_student_confidence": float(distilled_conf[index]),
            }
        )
        if len(rows) >= args.error_examples:
            return rows

    for index, text in enumerate(encoded_test.texts):
        if len(rows) >= args.error_examples:
            break
        category = category_for(index)
        if category is None:
            continue
        already_added = any(int(row["example_index"]) == index for row in rows)
        if already_added:
            continue
        rows.append(
            {
                "example_index": index,
                "category": category,
                "text": example_text(text),
                "true_label": LABEL_NAMES[int(labels[index])],
                "teacher_prediction": LABEL_NAMES[int(teacher_pred[index])],
                "hard_student_prediction": LABEL_NAMES[int(hard_pred[index])],
                "distilled_student_prediction": LABEL_NAMES[int(distilled_pred[index])],
                "teacher_confidence": float(teacher_conf[index]),
                "hard_student_confidence": float(hard_conf[index]),
                "distilled_student_confidence": float(distilled_conf[index]),
            }
        )
    return rows


def write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator=CSV_LINETERMINATOR
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, list):
            config[key] = [
                str(item) if isinstance(item, Path) else item for item in value
            ]
        else:
            config[key] = value
    return config


def fmt_pct(value: float) -> str:
    return f"{value * 100.0:.1f}\\%"


def fmt_ms(value: float) -> str:
    return f"{value:.2f} ms/example"


def fmt_mb(value: float) -> str:
    return f"{value:.1f} MB"


def fmt_ratio(value: float) -> str:
    return f"{value:.2f}x"


def make_result_values(
    result_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
) -> dict[str, str]:
    by_model = {row["model"]: row for row in result_rows}
    latency_by_model_batch = {
        (row["model"], int(row["batch_size"])): row for row in latency_rows
    }

    def latency_for(model_key: str, preferred_batch: int = 1) -> str:
        row = latency_by_model_batch.get((model_key, preferred_batch))
        if row is None:
            candidates = [item for item in latency_rows if item["model"] == model_key]
            row = candidates[0]
        return fmt_ms(float(row["mean_latency_ms_per_example"]))

    teacher_latency = float(
        latency_by_model_batch.get(("teacher", 1), latency_rows[0])[
            "mean_latency_ms_per_example"
        ]
    )
    distilled_latency = float(
        latency_by_model_batch.get(("distilled_student", 1), latency_rows[-1])[
            "mean_latency_ms_per_example"
        ]
    )
    hard_accuracy = float(by_model["hard_student"]["validation_accuracy"])
    distilled_accuracy = float(by_model["distilled_student"]["validation_accuracy"])
    accuracy_delta = distilled_accuracy - hard_accuracy
    speedup = teacher_latency / distilled_latency if distilled_latency > 0 else 0.0
    return {
        "RESULT_PLACEHOLDER_TEACHER_VALIDATION_ACCURACY": fmt_pct(
            float(by_model["teacher"]["validation_accuracy"])
        ),
        "RESULT_PLACEHOLDER_TEACHER_TEST_ACCURACY": fmt_pct(
            float(by_model["teacher"]["test_accuracy"])
        ),
        "RESULT_PLACEHOLDER_TEACHER_LATENCY": latency_for("teacher"),
        "RESULT_PLACEHOLDER_TEACHER_MODEL_SIZE": fmt_mb(
            float(by_model["teacher"]["model_size_mb"])
        ),
        "RESULT_PLACEHOLDER_HARD_LABEL_STUDENT_ACCURACY": fmt_pct(
            float(by_model["hard_student"]["validation_accuracy"])
        ),
        "RESULT_PLACEHOLDER_HARD_LABEL_STUDENT_TEST_ACCURACY": fmt_pct(
            float(by_model["hard_student"]["test_accuracy"])
        ),
        "RESULT_PLACEHOLDER_HARD_LABEL_STUDENT_LATENCY": latency_for("hard_student"),
        "RESULT_PLACEHOLDER_HARD_LABEL_STUDENT_MODEL_SIZE": fmt_mb(
            float(by_model["hard_student"]["model_size_mb"])
        ),
        "RESULT_PLACEHOLDER_DISTILLED_STUDENT_ACCURACY": fmt_pct(
            float(by_model["distilled_student"]["validation_accuracy"])
        ),
        "RESULT_PLACEHOLDER_DISTILLED_STUDENT_TEST_ACCURACY": fmt_pct(
            float(by_model["distilled_student"]["test_accuracy"])
        ),
        "RESULT_PLACEHOLDER_DISTILLED_STUDENT_LATENCY": latency_for(
            "distilled_student"
        ),
        "RESULT_PLACEHOLDER_DISTILLED_STUDENT_MODEL_SIZE": fmt_mb(
            float(by_model["distilled_student"]["model_size_mb"])
        ),
        "RESULT_PLACEHOLDER_DISTILLATION_TRADEOFF_TABLE": "\\texttt{nntrev-reference-distillation-results.csv}",
        "RESULT_PLACEHOLDER_TRAINING_CURVES": "\\texttt{nntrev-reference-training-curves.csv}",
        "RESULT_PLACEHOLDER_LATENCY_TABLE": "\\texttt{nntrev-reference-latency-table.csv}",
        "RESULT_PLACEHOLDER_THROUGHPUT_TABLE": "\\texttt{nntrev-reference-latency-table.csv}",
        "RESULT_PLACEHOLDER_PARAMETER_COUNT_TABLE": "\\texttt{nntrev-reference-model-footprint.csv}",
        "RESULT_PLACEHOLDER_MODEL_SIZE_TABLE": "\\texttt{nntrev-reference-model-footprint.csv}",
        "RESULT_PLACEHOLDER_PEAK_MEMORY_TABLE": "\\texttt{nntrev-reference-model-footprint.csv}",
        "RESULT_PLACEHOLDER_CONFUSION_MATRIX": "\\texttt{nntrev-reference-confusion-matrix.csv}",
        "RESULT_PLACEHOLDER_CALIBRATION_SUMMARY": "\\texttt{nntrev-reference-calibration-summary.csv}",
        "RESULT_PLACEHOLDER_STUDENT_VS_TEACHER_ERROR_EXAMPLES": "\\texttt{nntrev-reference-error-examples.csv}",
        "RESULT_PLACEHOLDER_FAILURE_CASES": "\\texttt{nntrev-reference-error-examples.csv}",
        "RESULT_SUMMARY_ACCURACY_DELTA": fmt_pct(accuracy_delta),
        "RESULT_SUMMARY_DISTILLED_SPEEDUP": fmt_ratio(speedup),
    }


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    deps: dict[str, Any],
    device: Any,
    dataset_sizes: dict[str, int],
) -> None:
    torch = deps["torch"]
    lines = [
        "Neural network training revisited reference experiment",
        f"timestamp_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "hostname: not-recorded",
        f"platform: {platform.platform()}",
        f"python: {platform.python_version()}",
        f"torch: {torch.__version__}",
        f"cuda_available: {torch.cuda.is_available()}",
        f"cuda_version: {getattr(torch.version, 'cuda', None)}",
        f"device: {device}",
        f"teacher_checkpoint: {args.teacher_checkpoint}",
        f"student_checkpoint: {args.student_checkpoint}",
        f"tokenizer_checkpoint: {args.tokenizer_checkpoint or args.student_checkpoint}",
        f"dataset: {args.dataset}",
        f"dataset_sizes: {dataset_sizes}",
        f"max_length: {args.max_length}",
        f"batch_size: {args.batch_size}",
        f"eval_batch_size: {args.eval_batch_size}",
        f"teacher_epochs: {args.teacher_epochs}",
        f"student_epochs: {args.student_epochs}",
        f"temperature: {args.temperature}",
        f"alpha: {args.alpha}",
        f"mixed_precision: {args.mixed_precision}",
        f"latency_warmup: {args.latency_warmup}",
        f"latency_repetitions: {args.latency_repetitions}",
        f"include_error_text: {args.include_error_text}",
        "command: "
        + " ".join(shlex.quote(part).replace(str(Path.home()), "<home>") for part in sys.argv),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plots(
    output_dir: Path,
    history: list[dict[str, Any]],
    confusion: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    deps: dict[str, Any],
) -> None:
    pn = deps["plotnine"]
    if pn is None:
        return
    display_names = {
        row["model"]: row.get("display_name", row["model"]) for row in result_rows
    }
    colors = {
        "teacher": PLOT_STYLE.BLUE,
        "hard_student": PLOT_STYLE.ORANGE,
        "distilled_student": PLOT_STYLE.GREEN,
    }

    def save_plot(
        plot: Any,
        filename: str,
        *,
        width: float,
        height: float,
        dpi: int = PLOT_STYLE.PLOT_DPI,
    ) -> None:
        plot.save(
            output_dir / filename,
            width=width,
            height=height,
            units="in",
            dpi=dpi,
            verbose=False,
        )

    try:
        training_records = [
            {
                "epoch": int(row["epoch"]),
                "display_name": display_names.get(row["model"], row["model"]),
                "validation_accuracy": float(row["validation_accuracy"]),
            }
            for row in history
        ]
        if training_records:
            plot = (
                pn["ggplot"](
                    pn["pd"].DataFrame.from_records(training_records),
                    pn["aes"](
                        "epoch",
                        "validation_accuracy",
                        color="display_name",
                        group="display_name",
                    ),
                )
                + pn["geom_line"](size=0.8)
                + pn["geom_point"](size=2.2)
                + pn["scale_color_manual"](
                    values=pn["style"].palette_for(display_names.values())
                )
                + pn["scale_y_continuous"](
                    labels=lambda values: [f"{value:.3f}" for value in values]
                )
                + pn["labs"](x="Epoch", y="Validation accuracy", color="model")
                + pn["style"].plot_theme(pn["p9"])
            )
            save_plot(
                plot, f"{ARTIFACT_PREFIX}-training-curves.png", width=6.0, height=4.0
            )
    except Exception:
        pass

    try:
        matrix = [[0 for _ in LABEL_NAMES] for _ in LABEL_NAMES]
        for row in confusion:
            true_index = LABEL_NAMES.index(row["true_label"])
            pred_index = LABEL_NAMES.index(row["predicted_label"])
            matrix[true_index][pred_index] = int(row["count"])
        largest = max(max(row) for row in matrix) or 1
        records = []
        for true_index, row in enumerate(matrix):
            for pred_index, value in enumerate(row):
                records.append(
                    {
                        "true_label": LABEL_NAMES[true_index],
                        "predicted_label": LABEL_NAMES[pred_index],
                        "count": value,
                        "label": str(value),
                        "text_color": "white" if value > largest * 0.55 else "#222222",
                    }
                )
        plot = (
            pn["ggplot"](
                pn["pd"].DataFrame.from_records(records),
                pn["aes"]("predicted_label", "true_label", fill="count"),
            )
            + pn["geom_tile"](color="white", size=0.35)
            + pn["geom_text"](pn["aes"](label="label", color="text_color"), size=8)
            + pn["scale_color_identity"]()
            + pn["scale_fill_gradient"](
                low=pn["style"].CONFUSION_LOW, high=pn["style"].CONFUSION_HIGH
            )
            + pn["scale_x_discrete"](limits=LABEL_NAMES)
            + pn["scale_y_discrete"](limits=list(reversed(LABEL_NAMES)))
            + pn["coord_fixed"]()
            + pn["labs"](x="Predicted label", y="True label", fill="examples")
            + pn["style"].plot_theme(pn["p9"])
            + pn["theme"](axis_text_x=pn["element_text"](rotation=45, ha="right"))
        )
        save_plot(
            plot, f"{ARTIFACT_PREFIX}-confusion-matrix.png", width=5.0, height=4.0
        )
    except Exception:
        pass

    try:
        single_latency = {
            row["model"]: float(row["mean_latency_ms_per_example"])
            for row in latency_rows
            if int(row["batch_size"]) == 1
        }
        records = []
        for row in result_rows:
            model_key = row["model"]
            if model_key not in single_latency:
                continue
            size_mb = float(row["model_size_mb"])
            display_name = display_names.get(model_key, model_key)
            records.append(
                {
                    "model": model_key,
                    "display_name": display_name,
                    "latency": single_latency[model_key],
                    "accuracy": float(row["test_accuracy"]) * 100.0,
                    "model_size_mb": size_mb,
                    "label": f"{display_name}\n{size_mb:.1f} MB",
                }
            )
        if records:
            fill_values = {
                row["display_name"]: colors.get(row["model"], "#777777")
                for row in records
            }
            plot = (
                pn["ggplot"](
                    pn["pd"].DataFrame.from_records(records),
                    pn["aes"](
                        "latency", "accuracy", fill="display_name", size="model_size_mb"
                    ),
                )
                + pn["geom_point"](color="black", alpha=0.85)
                + pn["geom_text"](
                    pn["aes"](label="label"), nudge_x=0.1, size=8, ha="left"
                )
                + pn["scale_fill_manual"](values=fill_values)
                + pn["scale_size_continuous"](range=(4, 13))
                + pn["labs"](
                    title="Accuracy, Latency, and Model Size Tradeoff",
                    x="Single-example latency (ms, lower is better)",
                    y="Test accuracy (%)",
                    fill="model",
                    size="model size (MB)",
                )
                + pn["style"].plot_theme(pn["p9"])
            )
            save_plot(
                plot,
                f"{ARTIFACT_PREFIX}-accuracy-latency-size.png",
                width=7.2,
                height=4.6,
            )
    except Exception:
        pass

    try:
        records = []
        compact_labels = {
            "teacher": "Teacher",
            "hard_student": "Hard-label\nstudent",
            "distilled_student": "Distilled\nstudent",
            "Teacher Transformer": "Teacher",
            "Hard-label student": "Hard-label\nstudent",
            "Distilled student": "Distilled\nstudent",
        }
        model_order = ["Teacher", "Hard-label\nstudent", "Distilled\nstudent"]
        series_order = ["Accuracy", "Mean confidence", "ECE"]
        for row in calibration_rows:
            name = display_names.get(row["model"], row["model"])
            plot_label = compact_labels.get(
                row["model"], compact_labels.get(name, name)
            )
            for panel, series, value in [
                ("Accuracy vs. Confidence", "Accuracy", float(row["accuracy"]) * 100.0),
                (
                    "Accuracy vs. Confidence",
                    "Mean confidence",
                    float(row["mean_confidence"]) * 100.0,
                ),
                (
                    "Expected Calibration Error",
                    "ECE",
                    float(row["expected_calibration_error"]) * 100.0,
                ),
            ]:
                records.append(
                    {
                        "display_name": plot_label,
                        "panel": panel,
                        "series": series,
                        "value": value,
                        "label": f"{value:.1f}",
                    }
                )
        if records:
            data = pn["pd"].DataFrame.from_records(records)
            data["panel"] = pn["pd"].Categorical(
                data["panel"],
                categories=["Accuracy vs. Confidence", "Expected Calibration Error"],
                ordered=True,
            )
            data["display_name"] = pn["pd"].Categorical(
                data["display_name"],
                categories=model_order,
                ordered=True,
            )
            data["series"] = pn["pd"].Categorical(
                data["series"],
                categories=series_order,
                ordered=True,
            )
            dodge = pn["position_dodge"](width=0.74)
            plot = (
                pn["ggplot"](data, pn["aes"]("display_name", "value", fill="series"))
                + pn["geom_col"](position=dodge, width=0.58, color="white", size=0.3)
                + pn["geom_text"](
                    pn["aes"](label="label"),
                    position=dodge,
                    va="bottom",
                    size=6.5,
                    color="#26323f",
                )
                + pn["facet_wrap"]("~panel", scales="free_y", nrow=1)
                + pn["scale_fill_manual"](
                    values={
                        "Accuracy": PLOT_STYLE.BLUE,
                        "Mean confidence": PLOT_STYLE.ORANGE,
                        "ECE": PLOT_STYLE.GRAY,
                    }
                )
                + pn["labs"](
                    title="Calibration Summary on the Test Split",
                    x="",
                    y="Percent",
                    fill="metric",
                )
                + pn["style"].plot_theme(pn["p9"], legend_position="top")
                + pn["theme"](
                    plot_title=pn["element_text"](size=13, weight="bold", ha="center"),
                    strip_text=pn["element_text"](size=10.5, weight="bold"),
                    strip_background=pn["element_rect"](
                        fill="#F3F6FA", color="#D9DEE7", size=0.5
                    ),
                    axis_title_y=pn["element_text"](size=9.5),
                    axis_text_x=pn["element_text"](rotation=0, ha="center", size=9.2),
                    axis_text_y=pn["element_text"](size=9),
                    legend_position="top",
                    legend_title=pn["element_blank"](),
                    panel_grid_major_x=pn["element_blank"](),
                    panel_grid_major_y=pn["element_line"](color="#D9DEE7", size=0.35),
                    panel_grid_minor=pn["element_blank"](),
                    panel_border=pn["element_rect"](
                        fill=None, color="#D9DEE7", size=0.5
                    ),
                    plot_background=pn["element_rect"](fill="white", color="white"),
                )
            )
            save_plot(
                plot, f"{ARTIFACT_PREFIX}-calibration-ece.png", width=7.4, height=4.1
            )
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_quick_settings(args)
    deps = load_dependencies(args)
    if deps is None:
        return 0
    if args.check_deps:
        print("All experiment dependencies imported.")
        return 0

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args, deps)
    torch = deps["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_splits = load_text_splits(args, deps)
    tokenizer_checkpoint = args.tokenizer_checkpoint or args.student_checkpoint
    tokenizer = deps["AutoTokenizer"].from_pretrained(
        tokenizer_checkpoint,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        local_files_only=args.local_files_only,
    )
    encoded = {
        name: encode_split(tokenizer, texts, labels, args)
        for name, (texts, labels) in text_splits.items()
    }
    loaders = make_loaders(encoded, args, deps)

    dataset_sizes = {name: len(split.labels) for name, split in encoded.items()}
    print(f"Dataset sizes: {dataset_sizes}", flush=True)
    print(f"Device: {device}; precision: {args.mixed_precision}", flush=True)

    histories: list[dict[str, Any]] = []
    all_latency_rows: list[dict[str, Any]] = []

    teacher = load_sequence_classifier(args.teacher_checkpoint, args, deps)
    teacher_history: list[dict[str, Any]] = []
    teacher_elapsed = 0.0
    teacher_peak_allocated, teacher_peak_reserved = None, None
    if args.teacher_epochs > 0:
        (
            teacher_history,
            teacher_elapsed,
            teacher_peak_allocated,
            teacher_peak_reserved,
        ) = train_model(
            teacher,
            loaders,
            args,
            deps,
            device,
            epochs=args.teacher_epochs,
            learning_rate=args.teacher_learning_rate,
            model_key="teacher",
            teacher_logits=None,
        )
        histories.extend(teacher_history)
    else:
        teacher.to(device)
        reset_peak_memory(deps, device)
        start = time.perf_counter()
        _ = evaluate_model(teacher, loaders["validation"], args, deps, device)
        teacher_elapsed = time.perf_counter() - start
        teacher_peak_allocated, teacher_peak_reserved = read_peak_memory(deps, device)

    print("Evaluating teacher and collecting logits.", flush=True)
    teacher_validation = evaluate_model(
        teacher, loaders["validation"], args, deps, device
    )
    teacher_test = evaluate_model(teacher, loaders["test"], args, deps, device)
    teacher_train_logits = collect_logits(
        teacher, loaders["train_eval"], args, deps, device
    )
    teacher_total_params, teacher_trainable_params = parameter_count(teacher)
    teacher_model_size = estimate_model_size_mb(
        teacher, output_dir, "teacher", deps, args.save_models
    )
    teacher_latency_rows = measure_latency(
        teacher,
        encoded["test"],
        args,
        deps,
        device,
        model_key="teacher",
        display_name="Teacher Transformer",
    )
    all_latency_rows.extend(teacher_latency_rows)
    teacher_result = ModelResult(
        key="teacher",
        display_name="Teacher Transformer",
        validation_accuracy=teacher_validation["accuracy"],
        validation_loss=teacher_validation["loss"],
        test_accuracy=teacher_test["accuracy"],
        test_loss=teacher_test["loss"],
        parameter_count=teacher_total_params,
        trainable_parameter_count=teacher_trainable_params,
        model_size_mb=teacher_model_size,
        training_time_seconds=teacher_elapsed,
        peak_memory_allocated_mb=teacher_peak_allocated,
        peak_memory_reserved_mb=teacher_peak_reserved,
        notes="task-finetuned teacher"
        if args.teacher_epochs == 0
        else "teacher fine-tuned in this run",
    )
    teacher.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Training hard-label student.", flush=True)
    seed_everything(args, deps)
    hard_student = load_sequence_classifier(args.student_checkpoint, args, deps)
    hard_history, hard_elapsed, hard_peak_allocated, hard_peak_reserved = train_model(
        hard_student,
        loaders,
        args,
        deps,
        device,
        epochs=args.student_epochs,
        learning_rate=args.student_learning_rate,
        model_key="hard_student",
        teacher_logits=None,
    )
    histories.extend(hard_history)
    hard_validation = evaluate_model(
        hard_student, loaders["validation"], args, deps, device
    )
    hard_test = evaluate_model(hard_student, loaders["test"], args, deps, device)
    hard_total_params, hard_trainable_params = parameter_count(hard_student)
    hard_model_size = estimate_model_size_mb(
        hard_student, output_dir, "hard-student", deps, args.save_models
    )
    hard_latency_rows = measure_latency(
        hard_student,
        encoded["test"],
        args,
        deps,
        device,
        model_key="hard_student",
        display_name="Hard-label student",
    )
    all_latency_rows.extend(hard_latency_rows)
    hard_result = ModelResult(
        key="hard_student",
        display_name="Hard-label student",
        validation_accuracy=hard_validation["accuracy"],
        validation_loss=hard_validation["loss"],
        test_accuracy=hard_test["accuracy"],
        test_loss=hard_test["loss"],
        parameter_count=hard_total_params,
        trainable_parameter_count=hard_trainable_params,
        model_size_mb=hard_model_size,
        training_time_seconds=hard_elapsed,
        peak_memory_allocated_mb=hard_peak_allocated,
        peak_memory_reserved_mb=hard_peak_reserved,
        notes="same student trained only on hard labels",
    )
    hard_student.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Training distilled student.", flush=True)
    seed_everything(args, deps)
    distilled_student = load_sequence_classifier(args.student_checkpoint, args, deps)
    (
        distilled_history,
        distilled_elapsed,
        distilled_peak_allocated,
        distilled_peak_reserved,
    ) = train_model(
        distilled_student,
        loaders,
        args,
        deps,
        device,
        epochs=args.student_epochs,
        learning_rate=args.student_learning_rate,
        model_key="distilled_student",
        teacher_logits=teacher_train_logits,
    )
    histories.extend(distilled_history)
    distilled_validation = evaluate_model(
        distilled_student, loaders["validation"], args, deps, device
    )
    distilled_test = evaluate_model(
        distilled_student, loaders["test"], args, deps, device
    )
    distilled_total_params, distilled_trainable_params = parameter_count(
        distilled_student
    )
    distilled_model_size = estimate_model_size_mb(
        distilled_student, output_dir, "distilled-student", deps, args.save_models
    )
    distilled_latency_rows = measure_latency(
        distilled_student,
        encoded["test"],
        args,
        deps,
        device,
        model_key="distilled_student",
        display_name="Distilled student",
    )
    all_latency_rows.extend(distilled_latency_rows)
    distilled_result = ModelResult(
        key="distilled_student",
        display_name="Distilled student",
        validation_accuracy=distilled_validation["accuracy"],
        validation_loss=distilled_validation["loss"],
        test_accuracy=distilled_test["accuracy"],
        test_loss=distilled_test["loss"],
        parameter_count=distilled_total_params,
        trainable_parameter_count=distilled_trainable_params,
        model_size_mb=distilled_model_size,
        training_time_seconds=distilled_elapsed,
        peak_memory_allocated_mb=distilled_peak_allocated,
        peak_memory_reserved_mb=distilled_peak_reserved,
        notes=f"student trained with alpha={args.alpha:g}, temperature={args.temperature:g}",
    )

    result_objects = [teacher_result, hard_result, distilled_result]
    result_rows = [
        {
            "model": result.key,
            "display_name": result.display_name,
            "validation_accuracy": result.validation_accuracy,
            "validation_loss": result.validation_loss,
            "test_accuracy": result.test_accuracy,
            "test_loss": result.test_loss,
            "parameter_count": result.parameter_count,
            "trainable_parameter_count": result.trainable_parameter_count,
            "model_size_mb": result.model_size_mb,
            "training_time_seconds": result.training_time_seconds,
            "peak_memory_allocated_mb": result.peak_memory_allocated_mb,
            "peak_memory_reserved_mb": result.peak_memory_reserved_mb,
            "notes": result.notes,
        }
        for result in result_objects
    ]

    footprint_rows = [
        {
            "model": row["model"],
            "display_name": row["display_name"],
            "parameter_count": row["parameter_count"],
            "trainable_parameter_count": row["trainable_parameter_count"],
            "model_size_mb": row["model_size_mb"],
            "peak_memory_allocated_mb": row["peak_memory_allocated_mb"],
            "peak_memory_reserved_mb": row["peak_memory_reserved_mb"],
        }
        for row in result_rows
    ]
    calibration_rows = [
        {
            "model": "teacher",
            "display_name": "Teacher Transformer",
            **calibration_summary(
                teacher_test["probabilities"],
                teacher_test["labels"],
                args.calibration_bins,
                deps,
            ),
        },
        {
            "model": "hard_student",
            "display_name": "Hard-label student",
            **calibration_summary(
                hard_test["probabilities"],
                hard_test["labels"],
                args.calibration_bins,
                deps,
            ),
        },
        {
            "model": "distilled_student",
            "display_name": "Distilled student",
            **calibration_summary(
                distilled_test["probabilities"],
                distilled_test["labels"],
                args.calibration_bins,
                deps,
            ),
        },
    ]
    confusion = confusion_rows(
        distilled_test["predictions"],
        distilled_test["labels"],
        "distilled_student",
        "Distilled student",
        deps,
    )
    error_examples = make_error_examples(
        encoded["test"], teacher_test, hard_test, distilled_test, args
    )
    result_values = make_result_values(result_rows, all_latency_rows)

    write_csv(output_dir / f"{ARTIFACT_PREFIX}-distillation-results.csv", result_rows)
    write_csv(output_dir / f"{ARTIFACT_PREFIX}-training-curves.csv", histories)
    write_csv(output_dir / f"{ARTIFACT_PREFIX}-latency-table.csv", all_latency_rows)
    write_csv(output_dir / f"{ARTIFACT_PREFIX}-model-footprint.csv", footprint_rows)
    write_csv(
        output_dir / f"{ARTIFACT_PREFIX}-calibration-summary.csv", calibration_rows
    )
    write_csv(output_dir / f"{ARTIFACT_PREFIX}-confusion-matrix.csv", confusion)
    write_csv(output_dir / f"{ARTIFACT_PREFIX}-error-examples.csv", error_examples)
    write_json(output_dir / f"{ARTIFACT_PREFIX}-result-values.json", result_values)
    write_json(
        output_dir / f"{ARTIFACT_PREFIX}-distillation-results.json",
        {
            "config": jsonable_config(args),
            "dataset_sizes": dataset_sizes,
            "results": result_rows,
            "latency": all_latency_rows,
            "calibration": calibration_rows,
            "result_values": result_values,
        },
    )
    write_metadata(
        output_dir / f"{ARTIFACT_PREFIX}-reference-results-metadata.txt",
        args,
        deps,
        device,
        dataset_sizes,
    )
    save_plots(
        output_dir,
        histories,
        confusion,
        result_rows,
        all_latency_rows,
        calibration_rows,
        deps,
    )

    print(
        json.dumps({"output_dir": str(output_dir), "results": result_rows}, indent=2),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
