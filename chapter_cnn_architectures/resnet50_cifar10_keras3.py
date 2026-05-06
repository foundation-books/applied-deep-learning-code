#!/usr/bin/env python3
"""Keras 3 ResNet-family CIFAR-10 training script for the CNN architectures chapter."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import os
import platform
import shlex
import sys
import time
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
CHAPTER_DIR = CODE_DIR
DEFAULT_FIGURE_DIR = CHAPTER_DIR / "artifacts" / "figures"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

MODEL_CONFIGS = {
    "resnet18": ("basic", [2, 2, 2, 2]),
    "resnet34": ("basic", [3, 4, 6, 3]),
    "resnet50": ("bottleneck", [3, 4, 6, 3]),
    "tiny-resnet": ("basic", [1, 1, 1, 1]),
}


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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ResNet-family Keras 3 model on CIFAR-10."
    )
    parser.add_argument(
        "--backend",
        choices=["tensorflow", "torch", "jax"],
        default=None,
        help="Force a Keras backend. KERAS_BACKEND wins when this is omitted.",
    )
    parser.add_argument("--epochs", type=positive_int, default=50)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--validation-size", type=positive_int, default=5000)
    parser.add_argument(
        "--limit-train",
        type=positive_int,
        help="Use only the first N training examples after the validation split.",
    )
    parser.add_argument(
        "--limit-val",
        type=positive_int,
        help="Use only the first N validation examples.",
    )
    parser.add_argument(
        "--limit-test",
        type=positive_int,
        help="Use only the first N test examples.",
    )
    parser.add_argument(
        "--stem",
        choices=["cifar", "imagenet"],
        default="cifar",
        help="Use a CIFAR-style 3x3 stride-1 stem or an ImageNet-style 7x7 stride-2 stem.",
    )
    parser.add_argument(
        "--model-variant",
        choices=list(MODEL_CONFIGS),
        default="resnet18",
        help="Choose a named ResNet variant. ResNet18 is the default student-scale model.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adamw"],
        default="sgd",
    )
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=nonnegative_float, default=5e-4)
    parser.add_argument(
        "--schedule",
        choices=["constant", "cosine"],
        default="cosine",
    )
    parser.add_argument(
        "--augmentation",
        choices=["none", "basic"],
        default="basic",
        help="The basic recipe uses random crop with padding plus horizontal flip.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional Keras cache directory. Sets KERAS_HOME before importing Keras.",
    )
    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        help="Use deterministic CIFAR-10-shaped synthetic data for smoke tests.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use tiny synthetic-friendly limits and one epoch for environment checks.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the held-out test set. Use only for a final selected or verification run.",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save history figures, confusion matrix, per-class accuracy, and metadata.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory for generated experiment artifacts.",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Only check imports, backend selection, and package summaries.",
    )
    parser.add_argument(
        "--allow-missing-deps",
        action="store_true",
        help="Return success with a warning when optional ML dependencies are absent.",
    )
    return parser.parse_args(argv)


def command_line(argv: list[str]) -> str:
    return " ".join(
        shlex.quote(part) for part in ["python", Path(__file__).name, *argv]
    )


def configure_backend(args: argparse.Namespace) -> str:
    if args.backend is not None:
        backend = args.backend
    elif "KERAS_BACKEND" in os.environ:
        backend = os.environ["KERAS_BACKEND"]
    else:
        backend = "tensorflow"
    os.environ["KERAS_BACKEND"] = backend
    return backend


def configure_data_cache(data_dir: Path | None) -> str:
    if data_dir is None:
        return os.environ.get("KERAS_HOME", "default")
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KERAS_HOME"] = str(data_dir.resolve())
    return os.environ["KERAS_HOME"]


def import_keras(allow_missing_deps: bool) -> Any:
    try:
        import keras  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local optional deps
        message = f"Could not import Keras/backend dependencies: {exc}"
        if allow_missing_deps:
            print(f"[WARN] {message}")
            return None
        raise SystemExit(message) from exc
    return keras


def package_version(name: str) -> str:
    try:
        return f"{name} {version(name)}"
    except PackageNotFoundError:
        return f"{name} not installed"


def backend_package_summary(backend: str) -> str:
    if backend == "tensorflow":
        return package_version("tensorflow")
    if backend == "torch":
        return package_version("torch")
    if backend == "jax":
        return f"{package_version('jax')}; {package_version('jaxlib')}"
    return "unknown backend"


def backend_device_summary(backend: str) -> str:
    if backend == "tensorflow":
        try:
            import tensorflow as tf  # type: ignore
        except Exception as exc:  # pragma: no cover
            return f"TensorFlow import failed: {exc}"
        return f"TensorFlow GPU devices: {len(tf.config.list_physical_devices('GPU'))}"
    if backend == "torch":
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            return f"torch import failed: {exc}"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        mps_available = bool(mps is not None and mps.is_available())
        return f"torch CUDA available: {torch.cuda.is_available()}; MPS available: {mps_available}"
    if backend == "jax":
        try:
            import jax  # type: ignore
        except Exception as exc:  # pragma: no cover
            return f"JAX import failed: {exc}"
        return "JAX devices: " + ", ".join(str(device) for device in jax.devices())
    return "No device summary available."


def apply_quick_defaults(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.epochs = 1
    args.validation_size = min(args.validation_size, 64)
    if args.model_variant != "tiny-resnet":
        args.model_variant = "tiny-resnet"
    if not args.synthetic_data:
        args.synthetic_data = True


def conv_bn_relu(
    keras: Any,
    x: Any,
    filters: int,
    kernel_size: int,
    *,
    strides: int = 1,
    name: str,
) -> Any:
    layers = keras.layers
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def bottleneck_block(
    keras: Any,
    x: Any,
    filters: int,
    *,
    strides: int,
    name: str,
) -> Any:
    layers = keras.layers
    shortcut = x
    out_channels = filters * 4

    x = conv_bn_relu(
        keras,
        x,
        filters,
        1,
        strides=1,
        name=f"{name}_1",
    )
    x = conv_bn_relu(
        keras,
        x,
        filters,
        3,
        strides=strides,
        name=f"{name}_2",
    )
    x = layers.Conv2D(
        out_channels,
        1,
        padding="same",
        use_bias=False,
        name=f"{name}_3_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_3_bn")(x)

    shortcut_channels = int(shortcut.shape[-1])
    if strides != 1 or shortcut_channels != out_channels:
        shortcut = layers.Conv2D(
            out_channels,
            1,
            strides=strides,
            padding="same",
            use_bias=False,
            name=f"{name}_proj_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([shortcut, x])
    return layers.ReLU(name=f"{name}_out")(x)


def basic_block(
    keras: Any,
    x: Any,
    filters: int,
    *,
    strides: int,
    name: str,
) -> Any:
    layers = keras.layers
    shortcut = x

    x = conv_bn_relu(
        keras,
        x,
        filters,
        3,
        strides=strides,
        name=f"{name}_1",
    )
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        name=f"{name}_2_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_2_bn")(x)

    shortcut_channels = int(shortcut.shape[-1])
    if strides != 1 or shortcut_channels != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=strides,
            padding="same",
            use_bias=False,
            name=f"{name}_proj_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([shortcut, x])
    return layers.ReLU(name=f"{name}_out")(x)


def build_resnet_model(keras: Any, args: argparse.Namespace) -> Any:
    layers = keras.layers
    block_type, block_counts = MODEL_CONFIGS[args.model_variant]
    block = bottleneck_block if block_type == "bottleneck" else basic_block
    inputs = keras.Input(shape=(32, 32, 3), name="image")
    x = inputs

    if args.augmentation == "basic":
        x = layers.ZeroPadding2D(4, name="augment_pad")(x)
        x = layers.RandomCrop(32, 32, seed=args.seed, name="augment_crop")(x)
        x = layers.RandomFlip("horizontal", seed=args.seed, name="augment_flip")(x)

    if args.stem == "imagenet":
        x = conv_bn_relu(
            keras,
            x,
            64,
            7,
            strides=2,
            name="stem_imagenet",
        )
        x = layers.MaxPooling2D(
            pool_size=3, strides=2, padding="same", name="stem_pool"
        )(x)
    else:
        x = conv_bn_relu(
            keras,
            x,
            64,
            3,
            strides=1,
            name="stem_cifar",
        )

    for stage_index, (filters, count) in enumerate(
        zip([64, 128, 256, 512], block_counts), start=2
    ):
        for block_index in range(count):
            stride = 2 if stage_index > 2 and block_index == 0 else 1
            x = block(
                keras,
                x,
                filters,
                strides=stride,
                name=f"conv{stage_index}_block{block_index + 1}",
            )

    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    outputs = layers.Dense(10, name="logits")(x)
    model_variant_name = args.model_variant.replace("-", "_")
    return keras.Model(
        inputs, outputs, name=f"cifar10_{model_variant_name}_{args.stem}"
    )


def build_optimizer(keras: Any, args: argparse.Namespace, steps_per_epoch: int) -> Any:
    learning_rate: float | Any = args.learning_rate
    if args.schedule == "cosine":
        learning_rate = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=max(1, steps_per_epoch * args.epochs),
        )

    if args.optimizer == "adamw":
        return keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=args.weight_decay,
        )

    try:
        return keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    except TypeError:
        return keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=args.momentum,
        )


def load_data(keras: Any, args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    if not args.synthetic_data:
        print("Dataset source: keras.datasets.cifar10")
        return keras.datasets.cifar10.load_data()

    import numpy as np

    train_count = max(args.validation_size + (args.limit_train or 128), 512)
    test_count = args.limit_test or (128 if args.quick else 1000)
    rng = np.random.default_rng(args.seed)
    x_train = rng.integers(0, 256, size=(train_count, 32, 32, 3), dtype="uint8")
    y_train = rng.integers(0, len(CLASS_NAMES), size=(train_count, 1), dtype="int64")
    x_test = rng.integers(0, 256, size=(test_count, 32, 32, 3), dtype="uint8")
    y_test = rng.integers(0, len(CLASS_NAMES), size=(test_count, 1), dtype="int64")
    print("Dataset source: synthetic CIFAR-10-shaped data")
    return (x_train, y_train), (x_test, y_test)


def split_data(
    x_full: Any, y_full: Any, args: argparse.Namespace
) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    if args.validation_size <= 0 or args.validation_size >= len(x_full):
        raise SystemExit(
            "--validation-size must be between 1 and the number of training examples"
        )

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(x_full))
    val_indices = indices[: args.validation_size]
    train_indices = indices[args.validation_size :]
    if args.quick:
        train_indices = train_indices[:128]
        val_indices = val_indices[:64]
    if args.limit_train is not None:
        train_indices = train_indices[: args.limit_train]
    if args.limit_val is not None:
        val_indices = val_indices[: args.limit_val]
    return (
        x_full[train_indices],
        y_full[train_indices],
        x_full[val_indices],
        y_full[val_indices],
    )


def history_rows(history: dict[str, list[float]]) -> list[dict[str, float | int]]:
    metric_names = [
        name
        for name in ["loss", "accuracy", "val_loss", "val_accuracy"]
        if name in history
    ]
    metric_names.extend(sorted(name for name in history if name not in metric_names))
    rows: list[dict[str, float | int]] = []
    for index in range(max((len(history[name]) for name in metric_names), default=0)):
        row: dict[str, float | int] = {"epoch": index + 1}
        for name in metric_names:
            if index < len(history[name]):
                row[name] = float(history[name][index])
        rows.append(row)
    return rows


def save_history(
    rows: list[dict[str, float | int]], output_dir: Path, prefix: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}-history.csv"
    fieldnames = list(rows[0]) if rows else ["epoch"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def import_plotnine() -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import pandas as pd
        import plotnine as p9
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Saving figures requires plotnine. Install with poetry install --with figures"
        ) from exc
    return {
        "pd": pd,
        "aes": p9.aes,
        "coord_fixed": p9.coord_fixed,
        "element_text": p9.element_text,
        "facet_wrap": p9.facet_wrap,
        "geom_col": p9.geom_col,
        "geom_line": p9.geom_line,
        "geom_point": p9.geom_point,
        "geom_text": p9.geom_text,
        "geom_tile": p9.geom_tile,
        "ggplot": p9.ggplot,
        "labs": p9.labs,
        "p9": p9,
        "scale_color_identity": p9.scale_color_identity,
        "scale_color_manual": p9.scale_color_manual,
        "scale_fill_gradient": p9.scale_fill_gradient,
        "scale_fill_manual": p9.scale_fill_manual,
        "scale_x_continuous": p9.scale_x_continuous,
        "scale_x_discrete": p9.scale_x_discrete,
        "scale_y_discrete": p9.scale_y_discrete,
        "style": PLOT_STYLE,
        "theme": p9.theme,
        "theme_minimal": p9.theme_minimal,
    }


def save_plot(
    plot: Any,
    path: Path,
    *,
    width: float,
    height: float,
    dpi: int = PLOT_STYLE.PLOT_DPI,
) -> Path:
    plot.save(path, width=width, height=height, units="in", dpi=dpi, verbose=False)
    return path


def row_metric_values(rows: list[dict[str, float | int]], *names: str) -> list[float]:
    for name in names:
        values = [float(row[name]) for row in rows if name in row]
        if values:
            return values
    return []


def save_training_curves(
    rows: list[dict[str, float | int]], output_dir: Path, prefix: str
) -> Path:
    pn = import_plotnine()
    path = output_dir / f"{prefix}-training-curves.png"
    if len(rows) == 1:
        row = rows[0]
        records = []
        for panel, values in [
            ("Loss after epoch 1", [float(row["loss"]), float(row["val_loss"])]),
            (
                "Accuracy after epoch 1",
                [float(row["accuracy"]), float(row["val_accuracy"])],
            ),
        ]:
            offset = max(values) * 0.045 if values else 0.02
            for split, value in zip(["train", "validation"], values):
                records.append(
                    {
                        "panel": panel,
                        "split": split,
                        "value": value,
                        "label": f"{value:.3f}",
                        "label_y": value + offset,
                    }
                )
        data = pn["pd"].DataFrame.from_records(records)
        data["panel"] = pn["pd"].Categorical(
            data["panel"],
            categories=["Loss after epoch 1", "Accuracy after epoch 1"],
            ordered=True,
        )
        plot = (
            pn["ggplot"](data, pn["aes"]("split", "value", fill="split"))
            + pn["geom_col"](width=0.55)
            + pn["geom_text"](pn["aes"](y="label_y", label="label"), size=8)
            + pn["facet_wrap"]("~panel", scales="free_y", nrow=1)
            + pn["scale_fill_manual"](
                values=pn["style"].palette_for(["train", "validation"])
            )
            + pn["labs"](
                title="CIFAR-10 ResNet one-epoch verification", x="", y="metric value"
            )
            + pn["style"].plot_theme(pn["p9"])
            + pn["theme"](legend_position="none")
        )
        return save_plot(plot, path, width=7.8, height=3.4)

    epochs = [int(row["epoch"]) for row in rows]
    records = []
    for panel, split, values in [
        ("Loss", "train", row_metric_values(rows, "loss", "train_loss")),
        ("Loss", "validation", row_metric_values(rows, "val_loss")),
        ("Accuracy", "train", row_metric_values(rows, "accuracy", "train_accuracy")),
        ("Accuracy", "validation", row_metric_values(rows, "val_accuracy")),
    ]:
        for epoch, value in zip(epochs, values):
            records.append(
                {"epoch": epoch, "panel": panel, "split": split, "value": value}
            )
    data = pn["pd"].DataFrame.from_records(records)
    data["panel"] = pn["pd"].Categorical(
        data["panel"], categories=["Loss", "Accuracy"], ordered=True
    )
    plot = (
        pn["ggplot"](data, pn["aes"]("epoch", "value", color="split", group="split"))
        + pn["geom_line"](size=0.8)
        + pn["geom_point"](size=2.2)
        + pn["facet_wrap"]("~panel", scales="free_y", nrow=1)
        + pn["scale_color_manual"](
            values=pn["style"].palette_for(["train", "validation"])
        )
        + pn["scale_x_continuous"](breaks=epochs)
        + pn["labs"](
            title="CIFAR-10 ResNet training", x="epoch", y="metric value", color="split"
        )
        + pn["style"].plot_theme(pn["p9"])
    )
    return save_plot(plot, path, width=8.8, height=3.4)


def confusion_matrix(y_true: Any, y_pred: Any) -> list[list[int]]:
    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label)][int(pred_label)] += 1
    return matrix


def save_confusion_csv(matrix: list[list[int]], output_dir: Path, prefix: str) -> Path:
    path = output_dir / f"{prefix}-confusion-matrix.csv"
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(["true_label", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *row])
    return path


def save_per_class_accuracy(
    matrix: list[list[int]], output_dir: Path, prefix: str
) -> Path:
    path = output_dir / f"{prefix}-per-class-accuracy.csv"
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(["class", "correct", "total", "accuracy"])
        for class_name, row_index in zip(CLASS_NAMES, range(len(CLASS_NAMES))):
            total = sum(matrix[row_index])
            correct = matrix[row_index][row_index]
            accuracy = correct / total if total else 0.0
            writer.writerow([class_name, correct, total, f"{accuracy:.6f}"])
    return path


def save_confusion_png(matrix: list[list[int]], output_dir: Path, prefix: str) -> Path:
    pn = import_plotnine()
    path = output_dir / f"{prefix}-confusion-matrix.png"
    largest = max(max(row) for row in matrix) or 1
    records = []
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            records.append(
                {
                    "true_label": CLASS_NAMES[row_index],
                    "predicted_label": CLASS_NAMES[col_index],
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
        + pn["geom_text"](pn["aes"](label="label", color="text_color"), size=6)
        + pn["scale_color_identity"]()
        + pn["scale_fill_gradient"](
            low=pn["style"].CONFUSION_LOW, high=pn["style"].CONFUSION_HIGH
        )
        + pn["scale_x_discrete"](limits=CLASS_NAMES)
        + pn["scale_y_discrete"](limits=list(reversed(CLASS_NAMES)))
        + pn["coord_fixed"]()
        + pn["labs"](
            title="CIFAR-10 validation-set confusion matrix",
            x="predicted label",
            y="true label",
            fill="examples",
        )
        + pn["style"].plot_theme(pn["p9"])
        + pn["theme"](axis_text_x=pn["element_text"](rotation=45, ha="right"))
    )
    return save_plot(plot, path, width=6.4, height=5.8)


def write_metadata(
    output_dir: Path,
    prefix: str,
    files: list[Path],
    *,
    args: argparse.Namespace,
    command: str,
    run_started_utc: str,
    backend: str,
    backend_package: str,
    device_summary: str,
    parameter_count: int,
    elapsed: float,
    val_acc: float,
    test_loss: float | None,
    test_acc: float | None,
) -> Path:
    path = output_dir / f"{prefix}-metadata.txt"
    path.write_text(
        "\n".join(
            [
                "CNN architectures ResNet CIFAR-10 Keras experiment",
                f"Generated by {Path(__file__).resolve().relative_to(REPO_ROOT)}",
                f"Run started UTC: {run_started_utc}",
                f"Command: {command}",
                f"Python: {platform.python_version()}",
                f"Matplotlib: {package_version('matplotlib')}",
                f"Plotnine: {package_version('plotnine')}",
                f"Keras backend: {backend}",
                f"Backend package: {backend_package}",
                f"Device summary: {device_summary}",
                f"Seed: {args.seed}",
                f"Model variant: {args.model_variant}",
                f"Stem: {args.stem}",
                f"Optimizer: {args.optimizer}",
                f"Learning rate: {args.learning_rate}",
                f"Schedule: {args.schedule}",
                f"Weight decay: {args.weight_decay}",
                f"Batch size: {args.batch_size}",
                f"Epochs: {args.epochs}",
                f"Augmentation: {args.augmentation}",
                f"Validation size: {args.validation_size}",
                f"Train limit: {args.limit_train}",
                f"Validation limit: {args.limit_val}",
                f"Test limit: {args.limit_test}",
                f"Parameter count: {parameter_count}",
                f"Elapsed seconds: {elapsed:.2f}",
                f"Final validation accuracy: {val_acc:.4f}",
                f"Test evaluation: {'enabled' if test_acc is not None else 'skipped'}",
                *(
                    [
                        f"Test loss: {test_loss:.4f}",
                        f"Test accuracy: {test_acc:.4f}",
                    ]
                    if test_loss is not None and test_acc is not None
                    else []
                ),
                "Generated files:",
                *[f"- {file.name}" for file in files],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    apply_quick_defaults(args)
    requested_backend = configure_backend(args)
    data_cache = configure_data_cache(args.data_dir)
    keras = import_keras(args.allow_missing_deps)
    if keras is None:
        return 0

    actual_backend = keras.backend.backend()
    backend_package = backend_package_summary(actual_backend)
    device_summary = backend_device_summary(actual_backend)
    run_started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    command = command_line(argv)

    print("Python:", platform.python_version())
    print("Keras:", keras.__version__)
    print("Requested backend:", requested_backend)
    print("Backend:", actual_backend)
    print("Backend package:", backend_package)
    print("Device summary:", device_summary)
    print("KERAS_HOME:", data_cache)
    print("Command:", command)
    print("Run started UTC:", run_started_utc)
    print(f"model_variant={args.model_variant}")
    print(f"stem={args.stem}")
    print(f"optimizer={args.optimizer}")
    print(f"schedule={args.schedule}")
    print(f"augmentation={args.augmentation}")

    if args.check_deps:
        print("Dependency check passed.")
        return 0

    keras.utils.set_random_seed(args.seed)
    (x_full, y_full), (x_test, y_test) = load_data(keras, args)
    x_full = x_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_full = y_full.squeeze().astype("int64")
    y_test = y_test.squeeze().astype("int64")
    x_train, y_train, x_val, y_val = split_data(x_full, y_full, args)
    if args.quick:
        x_test = x_test[:64]
        y_test = y_test[:64]
    if args.limit_test is not None:
        x_test = x_test[: args.limit_test]
        y_test = y_test[: args.limit_test]

    print("train:", x_train.shape, y_train.shape)
    print("val:", x_val.shape, y_val.shape)
    if args.evaluate_test:
        print("test:", x_test.shape, y_test.shape)
    else:
        print("test: skipped (--evaluate-test not set)")
    print("pixel range:", float(x_train.min()), float(x_train.max()))
    print("seed:", args.seed)
    print("batch_size:", args.batch_size)
    print("epochs:", args.epochs)
    print("validation_size:", args.validation_size)
    print("limit_train:", args.limit_train)
    print("limit_val:", args.limit_val)
    print("limit_test:", args.limit_test)

    model = build_resnet_model(keras, args)
    parameter_count = int(model.count_params())
    print(f"parameter_count={parameter_count}")
    steps_per_epoch = max(1, (len(x_train) + args.batch_size - 1) // args.batch_size)
    model.compile(
        optimizer=build_optimizer(keras, args, steps_per_epoch),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=2,
    )
    elapsed = time.perf_counter() - start

    val_loss = float(history.history["val_loss"][-1])
    val_acc = float(history.history["val_accuracy"][-1])
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"final_val_loss={val_loss:.4f} final_val_accuracy={val_acc:.4f}")
    test_loss: float | None = None
    test_acc: float | None = None
    if args.evaluate_test:
        raw_test_loss, raw_test_acc = model.evaluate(x_test, y_test, verbose=0)
        test_loss = float(raw_test_loss)
        test_acc = float(raw_test_acc)
        print(f"test_loss={test_loss:.4f} test_accuracy={test_acc:.4f}")
    else:
        print("test_evaluation=skipped")

    validation_logits = model.predict(x_val, batch_size=args.batch_size, verbose=0)
    validation_predicted = keras.ops.convert_to_numpy(
        keras.ops.argmax(validation_logits, axis=-1)
    )
    print(
        "validation predicted:", [CLASS_NAMES[int(i)] for i in validation_predicted[:8]]
    )
    print("validation true:", [CLASS_NAMES[int(i)] for i in y_val[:8]])

    if args.save_figures:
        prefix = f"{args.model_variant}-cifar10-keras"
        args.figure_dir.mkdir(parents=True, exist_ok=True)
        rows = history_rows(history.history)
        matrix = confusion_matrix(y_val, validation_predicted)
        written = [
            save_history(rows, args.figure_dir, prefix),
            save_training_curves(rows, args.figure_dir, prefix),
            save_confusion_csv(matrix, args.figure_dir, prefix),
            save_confusion_png(matrix, args.figure_dir, prefix),
            save_per_class_accuracy(matrix, args.figure_dir, prefix),
        ]
        written.append(
            write_metadata(
                args.figure_dir,
                prefix,
                written,
                args=args,
                command=command,
                run_started_utc=run_started_utc,
                backend=actual_backend,
                backend_package=backend_package,
                device_summary=device_summary,
                parameter_count=parameter_count,
                elapsed=elapsed,
                val_acc=val_acc,
                test_loss=test_loss,
                test_acc=test_acc,
            )
        )
        for path in written:
            print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
