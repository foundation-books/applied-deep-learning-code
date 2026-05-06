#!/usr/bin/env python3
"""Train IMDB sentiment baselines and RNNs with Keras 3."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

SHARED_CODE_DIR = Path(__file__).resolve().parents[1]
if str(SHARED_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_CODE_DIR))

from imdb_sentiment_shared import (  # noqa: E402
    build_vocabulary,
    load_optional_labeled_test_texts,
    load_training_texts,
    mean_token_length,
    stratified_split_count,
    vectorize_texts,
)


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


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an IMDB sentiment model using Keras 3.",
    )
    parser.add_argument(
        "--backend",
        choices=("tensorflow", "torch", "jax"),
        help="Set KERAS_BACKEND before importing Keras.",
    )
    parser.add_argument(
        "--model",
        choices=("average", "simple-rnn", "gru", "lstm"),
        default="lstm",
        help="Model family to train.",
    )
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--data-source",
        choices=("shared-imdb", "keras-imdb"),
        default="shared-imdb",
        help="Use the shared raw-text IMDB protocol or the legacy Keras integerized loader.",
    )
    parser.add_argument(
        "--data-dir",
        default="../chapter_embeddings/data",
        help="Directory containing shared IMDB data: aclImdb/ or labeledTrainData.tsv.",
    )
    parser.add_argument("--num-words", type=positive_int, default=10000)
    parser.add_argument("--max-length", type=positive_int, default=200)
    parser.add_argument("--embedding-dim", type=positive_int, default=64)
    parser.add_argument("--hidden-size", type=positive_int, default=64)
    parser.add_argument("--dropout", type=nonnegative_float, default=0.0)
    parser.add_argument("--recurrent-dropout", type=nonnegative_float, default=0.0)
    parser.add_argument("--clipnorm", type=nonnegative_float, default=1.0)
    parser.add_argument(
        "--early-stopping-patience",
        type=nonnegative_int,
        default=0,
        help="Stop after this many epochs without validation-loss improvement; 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=nonnegative_float,
        default=0.0,
        help="Minimum validation-loss change required by early stopping.",
    )
    parser.add_argument("--validation-size", type=positive_int, default=5000)
    parser.add_argument("--limit-train", type=positive_int)
    parser.add_argument("--limit-val", type=positive_int)
    parser.add_argument("--limit-test", type=positive_int)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run one epoch on small subsets for environment checks.",
    )
    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        help="Use synthetic padded token sequences instead of real IMDB data.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the held-out test set. Use only for the selected model.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts",
        help="Directory for saved JSON/CSV artifacts when --save-artifacts is set.",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save run metadata and training history.",
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
    args.limit_train = min(args.limit_train or 512, 512)
    args.limit_val = min(args.limit_val or 128, 128)
    args.limit_test = min(args.limit_test or 128, 128)
    args.max_length = min(args.max_length, 80)
    args.num_words = min(args.num_words, 2000)
    args.embedding_dim = min(args.embedding_dim, 16)
    args.hidden_size = min(args.hidden_size, 16)


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


def make_synthetic_data(args: argparse.Namespace):
    import numpy as np

    rng = np.random.default_rng(args.seed)

    def split(size: int):
        lengths = rng.integers(8, args.max_length + 1, size=size)
        x = np.zeros((size, args.max_length), dtype="int32")
        y = np.zeros((size,), dtype="int32")
        for row, length in enumerate(lengths):
            tokens = rng.integers(1, args.num_words, size=int(length), dtype="int32")
            x[row, : int(length)] = tokens
            y[row] = int(tokens[: int(length)].mean() > (args.num_words / 2))
        return x, y, lengths

    train_count = args.limit_train or 1024
    val_count = args.limit_val or 256
    test_count = args.limit_test or 256
    x_train, y_train, train_lengths = split(train_count)
    x_val, y_val, val_lengths = split(val_count)
    x_test, y_test, test_lengths = split(test_count)
    lengths = {
        "train_mean": float(train_lengths.mean()),
        "val_mean": float(val_lengths.mean()),
        "test_mean": float(test_lengths.mean()),
    }
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), lengths


def load_imdb_data(args: argparse.Namespace, keras):
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.imdb.load_data(
        num_words=args.num_words,
        seed=args.seed,
    )

    if args.validation_size >= len(x_train_full):
        raise ValueError("--validation-size must be smaller than the training set")

    x_train_raw = x_train_full[:-args.validation_size]
    y_train = y_train_full[:-args.validation_size]
    x_val_raw = x_train_full[-args.validation_size:]
    y_val = y_train_full[-args.validation_size:]

    x_train_raw = take_first(x_train_raw, args.limit_train)
    y_train = take_first(y_train, args.limit_train)
    x_val_raw = take_first(x_val_raw, args.limit_val)
    y_val = take_first(y_val, args.limit_val)
    x_test = take_first(x_test, args.limit_test)
    y_test = take_first(y_test, args.limit_test)

    lengths = {
        "train_mean": float(sum(len(x) for x in x_train_raw) / len(x_train_raw)),
        "val_mean": float(sum(len(x) for x in x_val_raw) / len(x_val_raw)),
        "test_mean": float(sum(len(x) for x in x_test) / len(x_test)),
    }

    pad = keras.utils.pad_sequences
    x_train = pad(
        x_train_raw,
        maxlen=args.max_length,
        padding="post",
        truncating="post",
        value=0,
    )
    x_val = pad(
        x_val_raw,
        maxlen=args.max_length,
        padding="post",
        truncating="post",
        value=0,
    )
    x_test = pad(
        x_test,
        maxlen=args.max_length,
        padding="post",
        truncating="post",
        value=0,
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), lengths


def resolve_data_dir(data_dir: str) -> Path:
    path = Path(data_dir)
    if path.is_absolute() or path.exists():
        return path
    script_relative = Path(__file__).resolve().parent / path
    if script_relative.exists():
        return script_relative
    return path


def load_shared_imdb_data(args: argparse.Namespace):
    import numpy as np

    data_dir = resolve_data_dir(args.data_dir)
    reviews, labels, resolved_source = load_training_texts(data_dir, "shared-imdb", np)
    args.resolved_data_source = resolved_source
    args.resolved_data_dir = str(data_dir)

    train_idx, val_idx = stratified_split_count(labels, args.validation_size, args.seed, np)
    train_idx = take_first(train_idx, args.limit_train)
    val_idx = take_first(val_idx, args.limit_val)

    train_reviews = [reviews[int(index)] for index in train_idx]
    val_reviews = [reviews[int(index)] for index in val_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]

    vocab = build_vocabulary(train_reviews, args.num_words)
    x_train = vectorize_texts(train_reviews, vocab, args.max_length, np)
    x_val = vectorize_texts(val_reviews, vocab, args.max_length, np)

    test_data = load_optional_labeled_test_texts(data_dir, "shared-imdb", np)
    if test_data is None:
        test_reviews: list[str] = []
        y_test = np.asarray([], dtype="float32")
        args.resolved_test_source = "none"
    else:
        test_reviews, y_test, resolved_test_source = test_data
        test_reviews = take_first(test_reviews, args.limit_test)
        y_test = take_first(y_test, args.limit_test)
        args.resolved_test_source = resolved_test_source
    x_test = vectorize_texts(test_reviews, vocab, args.max_length, np)

    lengths = {
        "train_mean": mean_token_length(train_reviews),
        "val_mean": mean_token_length(val_reviews),
        "test_mean": mean_token_length(test_reviews),
    }
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), lengths


def build_model(args: argparse.Namespace, keras, layers):
    inputs = keras.Input(shape=(args.max_length,), dtype="int32")
    x = layers.Embedding(
        input_dim=args.num_words,
        output_dim=args.embedding_dim,
        mask_zero=True,
        name="token_embedding",
    )(inputs)

    if args.model == "average":
        x = layers.GlobalAveragePooling1D(name="average_tokens")(x)
    elif args.model == "simple-rnn":
        x = layers.SimpleRNN(
            args.hidden_size,
            dropout=args.dropout,
            recurrent_dropout=args.recurrent_dropout,
            name="simple_rnn",
        )(x)
    elif args.model == "gru":
        x = layers.GRU(
            args.hidden_size,
            dropout=args.dropout,
            recurrent_dropout=args.recurrent_dropout,
            name="gru",
        )(x)
    elif args.model == "lstm":
        x = layers.LSTM(
            args.hidden_size,
            dropout=args.dropout,
            recurrent_dropout=args.recurrent_dropout,
            name="lstm",
        )(x)
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"Unsupported model: {args.model}")

    outputs = layers.Dense(1, name="sentiment_logit")(x)
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=args.clipnorm)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0)],
    )
    return model


def padding_ratio(x) -> float:
    total = int(x.size)
    if total == 0:
        return 0.0
    real = int((x != 0).sum())
    return 1.0 - (real / total)


def save_artifacts(args: argparse.Namespace, history, summary: dict) -> None:
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"imdb_{args.model.replace('-', '_')}_seed{args.seed}"
    history_path = artifact_dir / f"{prefix}_history.csv"
    summary_path = artifact_dir / f"{prefix}_summary.json"

    keys = sorted(history.history)
    with history_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", *keys])
        writer.writeheader()
        for epoch in range(len(next(iter(history.history.values())))):
            row = {"epoch": epoch + 1}
            for key in keys:
                row[key] = history.history[key][epoch]
            writer.writerow(row)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"history_csv={history_path}")
    print(f"summary_json={summary_path}")


def make_callbacks(args: argparse.Namespace, keras) -> list:
    if args.early_stopping_patience == 0:
        return []
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            restore_best_weights=True,
        )
    ]


def best_metric(history_values: list[float], *, maximize: bool) -> tuple[float, int]:
    indexed_values = list(enumerate(history_values, start=1))
    epoch, value = max(indexed_values, key=lambda item: item[1]) if maximize else min(
        indexed_values,
        key=lambda item: item[1],
    )
    return float(value), int(epoch)


def main() -> int:
    args = parse_args()
    apply_quick_defaults(args)

    try:
        keras, layers = import_keras(args)
    except RuntimeError as exc:
        if args.allow_missing_deps:
            print(f"[WARN] {exc}")
            return 0
        print(exc, file=sys.stderr)
        return 1

    print("Python:", platform.python_version())
    print("Keras:", keras.__version__)
    print("Backend:", keras.backend.backend())

    if args.check_deps:
        print("Keras dependency check: ok")
        return 0

    keras.utils.set_random_seed(args.seed)

    if args.synthetic_data:
        dataset_source = "synthetic"
        args.resolved_data_source = "synthetic"
        args.resolved_data_dir = None
        args.resolved_test_source = "synthetic"
        print("Dataset source: synthetic padded token sequences")
        (x_train, y_train), (x_val, y_val), (x_test, y_test), length_summary = (
            make_synthetic_data(args)
        )
    elif args.data_source == "keras-imdb":
        dataset_source = "keras_imdb"
        args.resolved_data_source = "keras_imdb"
        args.resolved_data_dir = None
        args.resolved_test_source = "keras_imdb"
        print("Dataset source: Keras IMDB")
        (x_train, y_train), (x_val, y_val), (x_test, y_test), length_summary = (
            load_imdb_data(args, keras)
        )
    else:
        print("Dataset source: shared IMDB raw text")
        (x_train, y_train), (x_val, y_val), (x_test, y_test), length_summary = (
            load_shared_imdb_data(args)
        )
        dataset_source = str(args.resolved_data_source)
        print("Resolved data dir:", args.resolved_data_dir)
        print("Resolved test source:", args.resolved_test_source)

    print("train:", x_train.shape, y_train.shape)
    print("val:", x_val.shape, y_val.shape)
    print("test:", x_test.shape, y_test.shape)
    train_padding_ratio = padding_ratio(x_train)
    val_padding_ratio = padding_ratio(x_val)
    test_padding_ratio = padding_ratio(x_test)
    print(f"train_padding_ratio={train_padding_ratio:.4f}")
    print(f"val_padding_ratio={val_padding_ratio:.4f}")
    print(f"test_padding_ratio={test_padding_ratio:.4f}")
    print(
        "mean_raw_lengths="
        f"train:{length_summary['train_mean']:.1f} "
        f"val:{length_summary['val_mean']:.1f} "
        f"test:{length_summary['test_mean']:.1f}"
    )

    model = build_model(args, keras, layers)
    model.summary()

    callbacks = make_callbacks(args, keras)
    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.perf_counter() - start

    final_val_loss = float(history.history["val_loss"][-1])
    final_val_accuracy = float(history.history["val_accuracy"][-1])
    best_val_loss, best_val_loss_epoch = best_metric(
        history.history["val_loss"],
        maximize=False,
    )
    best_val_accuracy, best_val_accuracy_epoch = best_metric(
        history.history["val_accuracy"],
        maximize=True,
    )
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"final_val_loss={final_val_loss:.4f}")
    print(f"final_val_accuracy={final_val_accuracy:.4f}")
    print(f"best_val_loss={best_val_loss:.4f} epoch={best_val_loss_epoch}")
    print(f"best_val_accuracy={best_val_accuracy:.4f} epoch={best_val_accuracy_epoch}")

    summary = {
        "model": args.model,
        "dataset_source": dataset_source,
        "data_source_requested": args.data_source,
        "data_dir": args.resolved_data_dir,
        "test_source": args.resolved_test_source,
        "backend": keras.backend.backend(),
        "python_version": platform.python_version(),
        "keras_version": keras.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "seed": args.seed,
        "num_words": args.num_words,
        "max_length": args.max_length,
        "embedding_dim": args.embedding_dim,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "recurrent_dropout": args.recurrent_dropout,
        "clipnorm": args.clipnorm,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.history["val_loss"]),
        "validation_size": args.validation_size,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "limit_test": args.limit_test,
        "params": int(model.count_params()),
        "train_padding_ratio": train_padding_ratio,
        "val_padding_ratio": val_padding_ratio,
        "test_padding_ratio": test_padding_ratio,
        "mean_raw_lengths": length_summary,
        "elapsed_seconds": elapsed,
        "final_val_loss": final_val_loss,
        "final_val_accuracy": final_val_accuracy,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_val_accuracy_epoch": best_val_accuracy_epoch,
        "evaluate_test": args.evaluate_test,
    }

    if args.evaluate_test:
        if len(y_test) == 0:
            raise ValueError(
                "--evaluate-test requires a labeled test split. Use data/aclImdb or --data-source keras-imdb."
            )
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"test_loss={test_loss:.4f}")
        print(f"test_accuracy={test_accuracy:.4f}")
        summary["test_loss"] = float(test_loss)
        summary["test_accuracy"] = float(test_accuracy)
    else:
        print("test_evaluation=skipped")

    if args.save_artifacts:
        save_artifacts(args, history, summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
