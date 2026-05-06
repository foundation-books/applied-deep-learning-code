#!/usr/bin/env python3
"""Keras 3 CIFAR-10 ConvNet baseline for the CNN basics chapter."""

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small Keras 3 ConvNet on CIFAR-10."
    )
    parser.add_argument(
        "--backend",
        choices=["tensorflow", "torch", "jax"],
        default=None,
        help=(
            "Force a Keras backend. If omitted, KERAS_BACKEND wins; otherwise "
            "the script defaults to tensorflow."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Optional Keras cache directory for downloaded datasets. When set, "
            "the script assigns KERAS_HOME before importing Keras."
        ),
    )
    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        help=(
            "Use deterministic CIFAR-10-shaped synthetic data. This is intended "
            "for smoke tests and does not measure real CIFAR-10 accuracy."
        ),
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save training curves and a confusion matrix to the chapter figures directory.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the held-out test set. Use only for a final selected run.",
    )
    parser.add_argument(
        "--save-image-grid",
        action="store_true",
        help=(
            "Also save a grid of CIFAR-10 misclassified examples. Use in distributed "
            "materials only after checking dataset image reuse policy."
        ),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory for generated figure files when --save-figures is used.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a tiny subset and one epoch for environment checks.",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Only check whether Keras and the selected backend can be imported.",
    )
    parser.add_argument(
        "--allow-missing-deps",
        action="store_true",
        help="Return success with a warning when optional ML dependencies are absent.",
    )
    return parser.parse_args(argv)


def configure_backend(args: argparse.Namespace) -> str:
    if args.backend is not None:
        backend = args.backend
    elif "KERAS_BACKEND" in os.environ:
        backend = os.environ["KERAS_BACKEND"]
    else:
        backend = choose_default_backend()

    os.environ["KERAS_BACKEND"] = backend
    return backend


def configure_data_cache(data_dir: Path | None) -> str:
    if data_dir is None:
        return os.environ.get("KERAS_HOME", "default")
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KERAS_HOME"] = str(data_dir.resolve())
    return os.environ["KERAS_HOME"]


def choose_default_backend() -> str:
    return "tensorflow"


def import_keras(allow_missing_deps: bool) -> Any:
    try:
        import keras  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        message = f"Could not import Keras/backend dependencies: {exc}"
        if allow_missing_deps:
            print(f"[WARN] {message}")
            return None
        raise SystemExit(message) from exc
    return keras


def backend_device_summary(backend: str) -> str:
    if backend == "torch":
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional deps
            return f"torch import failed: {exc}"
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        mps_available = bool(mps_backend is not None and mps_backend.is_available())
        cuda_available = bool(torch.cuda.is_available())
        if mps_available:
            try:
                from keras.src.backend.torch.core import get_device  # type: ignore
            except Exception:
                return "torch MPS available: yes"
            return f"torch MPS available: yes; Keras torch device: {get_device()}"
        return f"torch MPS available: no; CUDA available: {cuda_available}"

    if backend == "tensorflow":
        try:
            import tensorflow as tf  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional deps
            return f"TensorFlow import failed: {exc}"
        gpu_devices = tf.config.list_physical_devices("GPU")
        return f"TensorFlow GPU devices: {len(gpu_devices)}"

    if backend == "jax":
        try:
            import jax  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional deps
            return f"JAX import failed: {exc}"
        return "JAX devices: " + ", ".join(str(device) for device in jax.devices())

    return "No accelerator summary available."


def backend_package_summary(backend: str) -> str:
    if backend == "torch":
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional deps
            return f"torch import failed: {exc}"
        return f"torch {torch.__version__}"

    if backend == "tensorflow":
        try:
            import tensorflow as tf  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional deps
            return f"TensorFlow import failed: {exc}"
        return f"tensorflow {tf.__version__}"

    if backend == "jax":
        try:
            import jax  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional deps
            return f"JAX import failed: {exc}"
        return f"jax {jax.__version__}"

    return "No backend package summary available."


def command_line(argv: list[str]) -> str:
    parts = ["python", Path(__file__).name, *argv]
    return " ".join(shlex.quote(part) for part in parts)


def package_version(name: str) -> str:
    try:
        return f"{name} {version(name)}"
    except PackageNotFoundError:
        return f"{name} not installed"


def figure_package_summary() -> str:
    return "; ".join(
        package_version(name) for name in ("matplotlib", "plotnine", "pillow", "numpy")
    )


def build_model(keras: Any) -> Any:
    layers = keras.layers
    return keras.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10),
        ]
    )


def import_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional local deps
        raise SystemExit(
            "Saving figures requires matplotlib. Install the optional figure "
            "dependencies, for example: poetry install --with figures"
        ) from exc
    PLOT_STYLE.apply_matplotlib_style(plt)
    return plt


def import_plotnine() -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import pandas as pd
        import plotnine as p9
    except Exception as exc:  # pragma: no cover - depends on optional local deps
        raise SystemExit(
            "Saving chart figures requires plotnine. Install the optional figure "
            "dependencies, for example: poetry install --with figures"
        ) from exc
    return {
        "pd": pd,
        "aes": p9.aes,
        "coord_fixed": p9.coord_fixed,
        "element_text": p9.element_text,
        "facet_wrap": p9.facet_wrap,
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


def history_metric_names(history: dict[str, list[float]]) -> list[str]:
    preferred = ["loss", "accuracy", "val_loss", "val_accuracy"]
    names = [name for name in preferred if name in history]
    names.extend(sorted(name for name in history if name not in preferred))
    return names


def history_rows(history: dict[str, list[float]]) -> list[dict[str, float | int]]:
    metric_names = history_metric_names(history)
    epoch_count = max((len(history[name]) for name in metric_names), default=0)
    rows: list[dict[str, float | int]] = []
    for epoch_index in range(epoch_count):
        row: dict[str, float | int] = {"epoch": epoch_index + 1}
        for metric_name in metric_names:
            values = history[metric_name]
            if epoch_index < len(values):
                row[metric_name] = float(values[epoch_index])
        rows.append(row)
    return rows


def save_history_log(rows: list[dict[str, float | int]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cifar10-baseline-history.csv"
    metric_names = [name for name in rows[0] if name != "epoch"] if rows else []
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", *metric_names],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_history_log(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for raw_row in reader:
            rows.append(
                {
                    key: float(value)
                    for key, value in raw_row.items()
                    if value is not None and value != ""
                }
            )
    return rows


def metric_values(rows: list[dict[str, float]], metric_name: str) -> list[float]:
    return [row[metric_name] for row in rows if metric_name in row]


def save_training_curves(rows: list[dict[str, float]], output_dir: Path) -> Path:
    pn = import_plotnine()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cifar10-training-curves.png"

    epochs = [int(row["epoch"]) for row in rows]
    records = []
    for panel, split, values in [
        ("Loss", "train", metric_values(rows, "loss")),
        ("Loss", "validation", metric_values(rows, "val_loss")),
        ("Accuracy", "train", metric_values(rows, "accuracy")),
        ("Accuracy", "validation", metric_values(rows, "val_accuracy")),
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
            title="CIFAR-10 ConvNet baseline",
            x="epoch",
            y="metric value",
            color="split",
        )
        + pn["style"].plot_theme(pn["p9"])
    )
    return save_plot(plot, path, width=8.8, height=3.4)


def build_confusion_matrix(
    y_true: Any, y_pred: Any, class_count: int
) -> list[list[int]]:
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[int(true_label)][int(predicted_label)] += 1
    return matrix


def save_confusion_matrix(
    y_true: Any, y_pred: Any, class_names: list[str], output_dir: Path
) -> Path:
    pn = import_plotnine()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cifar10-confusion-matrix.png"
    matrix = build_confusion_matrix(y_true, y_pred, len(class_names))

    largest = max(max(row) for row in matrix) or 1
    records = []
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            records.append(
                {
                    "true_label": class_names[row_index],
                    "predicted_label": class_names[column_index],
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
        + pn["scale_x_discrete"](limits=class_names)
        + pn["scale_y_discrete"](limits=list(reversed(class_names)))
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


def save_misclassified_grid(
    images: Any,
    y_true: Any,
    y_pred: Any,
    class_names: list[str],
    output_dir: Path,
    count: int = 16,
) -> Path:
    plt = import_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cifar10-misclassified-grid.png"

    mistake_indices = [
        index
        for index, (true_label, pred_label) in enumerate(zip(y_true, y_pred))
        if int(true_label) != int(pred_label)
    ][:count]
    if not mistake_indices:
        mistake_indices = list(range(min(count, len(images))))

    columns = 4
    rows = (len(mistake_indices) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(7.2, 1.85 * rows), dpi=PLOT_STYLE.PLOT_DPI
    )
    flat_axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for axis, index in zip(flat_axes, mistake_indices):
        true_name = class_names[int(y_true[index])]
        pred_name = class_names[int(y_pred[index])]
        axis.imshow(images[index])
        axis.set_title(f"true: {true_name}\npred: {pred_name}", fontsize=7)
        axis.axis("off")

    for axis in flat_axes[len(mistake_indices) :]:
        axis.axis("off")

    figure.suptitle("Selected CIFAR-10 prediction mistakes")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def write_figure_metadata(
    output_dir: Path,
    files: list[Path],
    *,
    args: argparse.Namespace,
    command: str,
    run_started_utc: str,
    backend: str,
    keras_version: str,
    backend_package: str,
    device_summary: str,
    train_shape: tuple[int, ...],
    val_shape: tuple[int, ...],
    test_shape: tuple[int, ...],
    pixel_range: tuple[float, float],
    elapsed: float,
    val_loss: float,
    val_acc: float,
    test_loss: float | None,
    test_acc: float | None,
) -> Path:
    path = output_dir / "cifar10-baseline-figure-metadata.txt"
    relative_files = [file.name for file in files]
    path.write_text(
        "\n".join(
            [
                "CNN basics CIFAR-10 baseline figure generation",
                f"Generated by {Path(__file__).resolve().relative_to(REPO_ROOT)}",
                f"Run started UTC: {run_started_utc}",
                f"Command: {command}",
                f"Python: {platform.python_version()}",
                f"Keras: {keras_version}",
                f"Backend: {backend}",
                f"Backend package: {backend_package}",
                f"Figure package versions: {figure_package_summary()}",
                f"Device summary: {device_summary}",
                "Host: not-recorded",
                f"Platform: {platform.platform()}",
                f"Seed: {args.seed}",
                f"Batch size: {args.batch_size}",
                f"Epochs: {args.epochs}",
                f"Validation size: {args.validation_size}",
                f"Train shape: {train_shape}",
                f"Validation shape: {val_shape}",
                f"Test shape: {test_shape}",
                f"Pixel range: {pixel_range[0]:.4f} to {pixel_range[1]:.4f}",
                f"Synthetic data: {args.synthetic_data}",
                f"Test evaluation: {'enabled' if test_acc is not None else 'skipped'}",
                f"Elapsed seconds: {elapsed:.2f}",
                f"Final validation loss: {val_loss:.4f}",
                f"Final validation accuracy: {val_acc:.4f}",
                *(
                    [
                        f"Test loss: {test_loss:.4f}",
                        f"Test accuracy: {test_acc:.4f}",
                    ]
                    if test_loss is not None and test_acc is not None
                    else []
                ),
                "History log: cifar10-baseline-history.csv",
                "Error-analysis split: validation",
                "Generated files:",
                *[f"- {name}" for name in relative_files],
                (
                    "Image grid note: cifar10-misclassified-grid.png contains "
                    "CIFAR-10 dataset images and should be used in distributed "
                    "materials only after checking the dataset image reuse policy."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def load_data(keras: Any, args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    if not args.synthetic_data:
        print("Dataset source: keras.datasets.cifar10")
        return keras.datasets.cifar10.load_data()

    import numpy as np

    train_count = max(args.validation_size + 1024, 6000)
    test_count = 1000
    rng = np.random.default_rng(args.seed)
    x_train = rng.integers(
        0,
        256,
        size=(train_count, 32, 32, 3),
        dtype="uint8",
    )
    y_train = rng.integers(0, len(CLASS_NAMES), size=(train_count, 1), dtype="int64")
    x_test = rng.integers(
        0,
        256,
        size=(test_count, 32, 32, 3),
        dtype="uint8",
    )
    y_test = rng.integers(0, len(CLASS_NAMES), size=(test_count, 1), dtype="int64")
    print("Dataset source: synthetic CIFAR-10-shaped data")
    return (x_train, y_train), (x_test, y_test)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    selected_backend = configure_backend(args)
    data_cache = configure_data_cache(args.data_dir)
    keras = import_keras(args.allow_missing_deps)
    if keras is None:
        return 0

    run_started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    command = command_line(argv)
    actual_backend = keras.backend.backend()
    backend_package = backend_package_summary(actual_backend)
    device_summary = backend_device_summary(actual_backend)
    print("Python:", platform.python_version())
    print("Keras:", keras.__version__)
    print("Backend:", actual_backend)
    print("Requested backend:", selected_backend)
    print("Backend package:", backend_package)
    print("Device summary:", device_summary)
    print("KERAS_HOME:", data_cache)
    print("Command:", command)
    print("Run started UTC:", run_started_utc)

    if args.check_deps:
        print("Dependency check passed.")
        return 0

    if args.quick:
        args.epochs = 1
        train_limit = 1024
        val_limit = 256
        test_limit = 256
    else:
        train_limit = None
        val_limit = args.validation_size
        test_limit = None

    print("Seed:", args.seed)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Validation size:", args.validation_size)

    keras.utils.set_random_seed(args.seed)

    (x_train_full, y_train_full), (x_test, y_test) = load_data(keras, args)
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train_full = y_train_full.squeeze().astype("int64")
    y_test = y_test.squeeze().astype("int64")

    if args.validation_size <= 0 or args.validation_size >= len(x_train_full):
        raise SystemExit("--validation-size must be between 1 and 49999")

    x_train = x_train_full[: -args.validation_size]
    y_train = y_train_full[: -args.validation_size]
    x_val = x_train_full[-args.validation_size :]
    y_val = y_train_full[-args.validation_size :]

    if args.quick:
        x_train = x_train[:train_limit]
        y_train = y_train[:train_limit]
        x_val = x_val[:val_limit]
        y_val = y_val[:val_limit]
        x_test = x_test[:test_limit]
        y_test = y_test[:test_limit]

    print("train:", x_train.shape, y_train.shape)
    print("val:", x_val.shape, y_val.shape)
    print("test:", x_test.shape, y_test.shape)
    pixel_range = (float(x_train.min()), float(x_train.max()))
    print("pixel range:", pixel_range[0], pixel_range[1])

    model = build_model(keras)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()

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

    test_loss: float | None = None
    test_acc: float | None = None
    if args.evaluate_test:
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    val_loss = float(history.history["val_loss"][-1])
    val_acc = float(history.history["val_accuracy"][-1])
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"final_val_loss={val_loss:.4f} final_val_accuracy={val_acc:.4f}")
    if test_loss is not None and test_acc is not None:
        print(f"test_loss={test_loss:.4f} test_accuracy={test_acc:.4f}")
    else:
        print("test_evaluation=skipped")

    validation_logits = model.predict(x_val, batch_size=args.batch_size, verbose=0)
    validation_probs = keras.ops.softmax(validation_logits, axis=-1)
    validation_predicted = keras.ops.convert_to_numpy(
        keras.ops.argmax(validation_probs, axis=-1)
    )
    print(
        "validation predicted:",
        [CLASS_NAMES[int(i)] for i in validation_predicted[:8]],
    )
    print("validation true:", [CLASS_NAMES[int(i)] for i in y_val[:8]])

    if args.save_figures:
        history_path = save_history_log(history_rows(history.history), args.figure_dir)
        logged_history = load_history_log(history_path)
        written = [
            history_path,
            save_training_curves(logged_history, args.figure_dir),
            save_confusion_matrix(
                y_val,
                validation_predicted,
                CLASS_NAMES,
                args.figure_dir,
            ),
        ]
        if args.save_image_grid:
            written.append(
                save_misclassified_grid(
                    x_val,
                    y_val,
                    validation_predicted,
                    CLASS_NAMES,
                    args.figure_dir,
                )
            )
        written.append(
            write_figure_metadata(
                args.figure_dir,
                written,
                args=args,
                command=command,
                run_started_utc=run_started_utc,
                backend=actual_backend,
                keras_version=keras.__version__,
                backend_package=backend_package,
                device_summary=device_summary,
                train_shape=tuple(int(dim) for dim in x_train.shape),
                val_shape=tuple(int(dim) for dim in x_val.shape),
                test_shape=tuple(int(dim) for dim in x_test.shape),
                pixel_range=pixel_range,
                elapsed=elapsed,
                val_loss=val_loss,
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
