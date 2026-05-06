#!/usr/bin/env python3
"""Train the MNIST MLP from the chapter with Keras 3."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time


INSTALL_HELP = """Install Keras plus a backend before training. Examples:
  poetry install --with keras-tensorflow
  poetry install --with keras-torch
  poetry install --with keras-jax
"""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the chapter MNIST MLP using Keras 3.",
    )
    parser.add_argument(
        "--backend",
        choices=("tensorflow", "torch", "jax"),
        help="Set KERAS_BACKEND before importing Keras.",
    )
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--limit-train",
        type=positive_int,
        help="Use only the first N training examples.",
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
        "--quick",
        action="store_true",
        help="Run one epoch on a small subset for environment checks.",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check whether Keras can import with the selected backend.",
    )
    parser.add_argument(
        "--allow-missing-deps",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def apply_quick_defaults(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.epochs = 1
    args.limit_train = min(args.limit_train or 1024, 1024)
    args.limit_val = min(args.limit_val or 256, 256)
    args.limit_test = min(args.limit_test or 256, 256)


def import_keras(args: argparse.Namespace):
    if args.backend:
        os.environ["KERAS_BACKEND"] = args.backend
    else:
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")

    try:
        import keras
        from keras import layers
    except Exception as exc:  # pragma: no cover - depends on local packages.
        backend = os.environ.get("KERAS_BACKEND", "tensorflow")
        raise RuntimeError(
            f"Could not import Keras with backend {backend!r}: {exc}\n"
            f"{INSTALL_HELP}"
        ) from exc

    return keras, layers


def take_first(values, limit: int | None):
    if limit is None:
        return values
    return values[:limit]


def main() -> int:
    args = parse_args()
    apply_quick_defaults(args)

    try:
        keras, layers = import_keras(args)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 0 if args.check_deps and args.allow_missing_deps else 1

    print("Python:", platform.python_version())
    print("Keras:", keras.__version__)
    print("Backend:", keras.backend.backend())

    if args.check_deps:
        print("Keras dependency check: ok")
        return 0

    keras.utils.set_random_seed(args.seed)

    (x_train_full, y_train_full), (x_test, y_test) = (
        keras.datasets.mnist.load_data()
    )

    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = take_first(x_train_full[:-10000], args.limit_train)
    y_train = take_first(y_train_full[:-10000], args.limit_train)
    x_val = take_first(x_train_full[-10000:], args.limit_val)
    y_val = take_first(y_train_full[-10000:], args.limit_val)
    x_test = take_first(x_test, args.limit_test)
    y_test = take_first(y_test, args.limit_test)

    print("train:", x_train.shape, y_train.shape)
    print("val:", x_val.shape, y_val.shape)
    print("test:", x_test.shape, y_test.shape)

    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()

    start = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    elapsed = time.perf_counter() - start

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"test_loss={test_loss:.4f} test_accuracy={test_acc:.4f}")

    logits = model.predict(x_test[:8], verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    predicted = keras.ops.argmax(probs, axis=-1)
    print("predicted:", keras.ops.convert_to_numpy(predicted))
    print("true:", y_test[:8])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
