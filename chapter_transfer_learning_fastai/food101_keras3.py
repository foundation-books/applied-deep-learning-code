#!/usr/bin/env python3
"""Keras 3 Food-101 transfer-learning companion script."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any


INSTALL_HELP = """Install Keras plus a backend before running this script. Examples:
  poetry install --with keras-tensorflow
  poetry install --with keras-torch
  poetry install --with keras-jax
"""


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


def valid_pct(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or smoke-test a Keras 3 Food-101 transfer-learning model."
    )
    parser.add_argument(
        "--backend",
        choices=["tensorflow", "torch", "jax"],
        help="Set KERAS_BACKEND before importing Keras.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Food-101 root containing images/ and train.txt/test.txt or meta/*.txt.",
    )
    parser.add_argument("--image-size", type=positive_int, default=224)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--valid-pct", type=valid_pct, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-epochs", type=nonnegative_int, default=1)
    parser.add_argument("--epochs", type=nonnegative_int, default=5)
    parser.add_argument("--head-lr", type=positive_float, default=3e-4)
    parser.add_argument("--fine-tune-lr", type=positive_float, default=1e-5)
    parser.add_argument(
        "--limit-train",
        type=positive_int,
        help="Use only the first N training examples after splitting.",
    )
    parser.add_argument(
        "--limit-val",
        type=positive_int,
        help="Use only the first N validation examples after splitting.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use deterministic synthetic images instead of reading Food-101.",
    )
    parser.add_argument(
        "--smoke-pretrained",
        action="store_true",
        help="Allow ImageNet weight download during smoke mode.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize ResNet50 randomly instead of downloading ImageNet weights.",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable Keras random augmentation layers.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run one tiny synthetic frozen-head epoch for environment checks.",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Only check Keras import, backend selection, and package summaries.",
    )
    parser.add_argument(
        "--allow-missing-deps",
        action="store_true",
        help="Return success when optional Keras dependencies are absent.",
    )
    return parser.parse_args(argv)


def apply_quick_defaults(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.smoke = True
    args.no_pretrained = True
    args.image_size = min(args.image_size, 64)
    args.batch_size = min(args.batch_size, 4)
    args.freeze_epochs = 1
    args.epochs = 0
    args.limit_train = min(args.limit_train or 8, 8)
    args.limit_val = min(args.limit_val or 4, 4)


def configure_backend(args: argparse.Namespace) -> str:
    backend = args.backend or os.environ.get("KERAS_BACKEND") or "tensorflow"
    os.environ["KERAS_BACKEND"] = backend
    return backend


def import_keras() -> Any:
    try:
        import keras  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional packages.
        backend = os.environ.get("KERAS_BACKEND", "tensorflow")
        raise RuntimeError(
            f"Could not import Keras with backend {backend!r}: {exc}\n{INSTALL_HELP}"
        ) from exc
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


def candidate_food101_roots() -> list[Path]:
    return [
        Path.cwd() / "food-101",
        Path.cwd().parent / "food-101",
        Path.home() / ".fastai" / "data" / "food-101",
        Path.home() / ".keras" / "datasets" / "food-101",
    ]


def split_file(root: Path, split_name: str) -> Path:
    for candidate in [root / f"{split_name}.txt", root / "meta" / f"{split_name}.txt"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {split_name}.txt under {root} or {root / 'meta'}."
    )


def resolve_data_root(data_root: Path | None) -> Path:
    if data_root is not None:
        root = data_root.expanduser().resolve()
        if not (root / "images").is_dir():
            raise FileNotFoundError(f"{root} does not contain an images/ directory.")
        split_file(root, "train")
        return root

    for root in candidate_food101_roots():
        if (root / "images").is_dir():
            try:
                split_file(root, "train")
            except FileNotFoundError:
                continue
            return root.resolve()

    roots = "\n  ".join(str(path) for path in candidate_food101_roots())
    raise FileNotFoundError(
        "Food-101 was not found. Pass --data-root after downloading Food-101. "
        f"Checked:\n  {roots}"
    )


def read_split_entries(root: Path, split_name: str) -> list[str]:
    txt = split_file(root, split_name)
    return [line.strip() for line in txt.read_text(encoding="utf-8").splitlines() if line.strip()]


def class_name(entry: str) -> str:
    return entry.split("/", maxsplit=1)[0]


def load_food101_records(
    root: Path,
    *,
    seed: int,
    valid_pct_value: float,
    limit_train: int | None,
    limit_val: int | None,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[str], int]:
    import numpy as np

    train_entries = read_split_entries(root, "train")
    test_entries = read_split_entries(root, "test")
    class_names = sorted({class_name(entry) for entry in [*train_entries, *test_entries]})
    label_to_index = {name: index for index, name in enumerate(class_names)}

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_entries))
    valid_count = max(1, int(round(len(train_entries) * valid_pct_value)))
    valid_indices = indices[:valid_count]
    train_indices = indices[valid_count:]
    if limit_train is not None:
        train_indices = train_indices[:limit_train]
    if limit_val is not None:
        valid_indices = valid_indices[:limit_val]

    def record(entry: str) -> tuple[Path, int]:
        return root / "images" / f"{entry}.jpg", label_to_index[class_name(entry)]

    train_records = [record(train_entries[index]) for index in train_indices]
    val_records = [record(train_entries[index]) for index in valid_indices]
    return train_records, val_records, class_names, len(test_entries)


def make_image_dataset_class(keras: Any) -> type:
    class ImagePathDataset(keras.utils.PyDataset):  # type: ignore[name-defined]
        def __init__(
            self,
            records: list[tuple[Path, int]],
            *,
            image_size: int,
            batch_size: int,
            shuffle: bool,
            seed: int,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            import numpy as np

            self.records = records
            self.image_size = image_size
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.rng = np.random.default_rng(seed)
            self.indices = np.arange(len(records))
            if self.shuffle:
                self.rng.shuffle(self.indices)

        def __len__(self) -> int:
            return math.ceil(len(self.records) / self.batch_size)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            import numpy as np
            from PIL import Image, ImageOps

            try:
                resample = Image.Resampling.BILINEAR
            except AttributeError:  # pragma: no cover - Pillow compatibility.
                resample = Image.BILINEAR

            batch_indices = self.indices[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            images = []
            labels = []
            for record_index in batch_indices:
                image_path, label = self.records[int(record_index)]
                with Image.open(image_path) as image:
                    image = ImageOps.fit(
                        image.convert("RGB"),
                        (self.image_size, self.image_size),
                        method=resample,
                    )
                    images.append(np.asarray(image, dtype="float32"))
                labels.append(label)
            return np.stack(images), np.asarray(labels, dtype="int64")

        def on_epoch_end(self) -> None:
            if self.shuffle:
                self.rng.shuffle(self.indices)

    return ImagePathDataset


def synthetic_arrays(args: argparse.Namespace) -> tuple[tuple[Any, Any], tuple[Any, Any], list[str]]:
    import numpy as np

    train_count = args.limit_train or 12
    val_count = args.limit_val or 6
    class_names = ["apple_pie", "pizza", "ramen"]
    rng = np.random.default_rng(args.seed)
    x_train = rng.integers(
        0,
        256,
        size=(train_count, args.image_size, args.image_size, 3),
        dtype="uint8",
    ).astype("float32")
    y_train = np.arange(train_count, dtype="int64") % len(class_names)
    x_val = rng.integers(
        0,
        256,
        size=(val_count, args.image_size, args.image_size, 3),
        dtype="uint8",
    ).astype("float32")
    y_val = np.arange(val_count, dtype="int64") % len(class_names)
    return (x_train, y_train), (x_val, y_val), class_names


def build_model(keras: Any, args: argparse.Namespace, num_classes: int) -> tuple[Any, Any]:
    from keras.applications import ResNet50  # type: ignore
    from keras.applications.resnet import preprocess_input  # type: ignore

    layers = keras.layers
    weights = None if args.no_pretrained or (args.smoke and not args.smoke_pretrained) else "imagenet"
    base_model = ResNet50(
        include_top=False,
        weights=weights,
        input_shape=(args.image_size, args.image_size, 3),
        pooling="avg",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(args.image_size, args.image_size, 3), name="image")
    x = inputs
    if not args.no_augmentation:
        x = layers.RandomFlip("horizontal", seed=args.seed, name="augment_flip")(x)
        x = layers.RandomRotation(0.05, seed=args.seed, name="augment_rotate")(x)
        x = layers.RandomZoom(0.10, seed=args.seed, name="augment_zoom")(x)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.Dropout(0.2, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, name="food101_logits")(x)
    model = keras.Model(inputs, outputs, name="food101_keras3_transfer")
    return model, base_model


def compile_model(keras: Any, model: Any, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


def count_params(weights: list[Any]) -> int:
    total = 0
    for weight in weights:
        shape = tuple(int(dim) for dim in weight.shape)
        total += math.prod(shape)
    return total


def print_model_counts(model: Any) -> None:
    total = count_params(model.weights)
    trainable = count_params(model.trainable_weights)
    print(f"parameters_total={total}")
    print(f"parameters_trainable={trainable}")
    print(f"parameters_frozen={total - trainable}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_quick_defaults(args)
    if args.image_size < 32:
        raise SystemExit("--image-size must be at least 32 for Keras ResNet50.")

    backend = configure_backend(args)
    try:
        keras = import_keras()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 0 if args.check_deps and args.allow_missing_deps else 1

    print("Python:", platform.python_version())
    print("Keras:", keras.__version__)
    print("Backend:", keras.backend.backend())
    print("Backend package:", backend_package_summary(backend))

    if args.check_deps:
        print("Keras dependency check: ok")
        return 0

    keras.utils.set_random_seed(args.seed)

    if args.smoke:
        train_data, val_data, class_names = synthetic_arrays(args)
        print("Dataset source: synthetic Food-101-shaped smoke data")
        print("Reserved test split: not used in smoke mode")
    else:
        root = resolve_data_root(args.data_root)
        train_records, val_records, class_names, test_count = load_food101_records(
            root,
            seed=args.seed,
            valid_pct_value=args.valid_pct,
            limit_train=args.limit_train,
            limit_val=args.limit_val,
        )
        dataset_cls = make_image_dataset_class(keras)
        train_data = dataset_cls(
            train_records,
            image_size=args.image_size,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
        )
        val_data = dataset_cls(
            val_records,
            image_size=args.image_size,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
        )
        print(f"Dataset source: {root}")
        print(f"Training records: {len(train_records)}")
        print(f"Validation records: {len(val_records)}")
        print(f"Reserved official test records: {test_count}")

    print(f"Classes: {len(class_names)}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")

    model, base_model = build_model(keras, args, len(class_names))
    compile_model(keras, model, args.head_lr)
    print_model_counts(model)

    start = time.perf_counter()
    if args.freeze_epochs > 0:
        print(f"Stage: frozen head for {args.freeze_epochs} epoch(s)")
        model.fit(train_data, validation_data=val_data, epochs=args.freeze_epochs)
    else:
        print("Stage: frozen head skipped")

    if args.epochs > 0:
        print(f"Stage: unfrozen fine-tuning for {args.epochs} epoch(s)")
        base_model.trainable = True
        compile_model(keras, model, args.fine_tune_lr)
        print_model_counts(model)
        model.fit(train_data, validation_data=val_data, epochs=args.epochs)
    else:
        print("Stage: unfrozen fine-tuning skipped")

    elapsed = time.perf_counter() - start
    val_loss, val_acc = model.evaluate(val_data, verbose=0)
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"validation_loss={val_loss:.4f} validation_accuracy={val_acc:.4f}")
    print("official_test_evaluation=reserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
