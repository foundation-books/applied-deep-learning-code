#!/usr/bin/env python3
"""Train a bag-of-embeddings sentiment model for IMDB movie reviews.

This script is intentionally small enough for a course homework setting. It
trains the chapter's required architecture:

    tokenizer -> Embedding -> GlobalAveragePooling1D -> Dense logit

The canonical homework protocol uses the public ACL IMDB raw-text data from
Maas et al. Use ``data/aclImdb`` or a converted ``labeledTrainData.tsv`` in
``chapter_embeddings/data``. Kaggle's Bag of Words Meets Bags of Popcorn
TSV files remain compatible for optional submissions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

SHARED_CODE_DIR = Path(__file__).resolve().parents[1]
if str(SHARED_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_CODE_DIR))

from imdb_sentiment_shared import (  # noqa: E402
    build_vocabulary,
    clean_and_tokenize,
    load_training_texts,
    make_synthetic_dataset,
    stratified_split,
    vectorize_texts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing shared IMDB data: aclImdb/ or labeledTrainData.tsv.",
    )
    parser.add_argument(
        "--data-source",
        choices=("auto", "shared-imdb", "acl-imdb", "kaggle-tsv"),
        default="auto",
        help="Real-data source to load. Auto accepts labeledTrainData.tsv or aclImdb/.",
    )
    parser.add_argument(
        "--backend",
        choices=("tensorflow", "torch", "jax"),
        default="tensorflow",
        help="Keras 3 backend to request before importing keras.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--vocab-size", type=int, default=20_000)
    parser.add_argument("--max-length", type=int, default=400)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small configuration for fast local checks.",
    )
    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        help="Use a tiny generated sentiment dataset instead of real IMDB files.",
    )
    parser.add_argument(
        "--make-submission",
        action="store_true",
        help="Create an optional Kaggle submission from testData.tsv after training.",
    )
    parser.add_argument(
        "--submission-mode",
        choices=("label", "probability"),
        default="label",
        help="Write 0/1 labels or probabilities in the submission column.",
    )
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--run-tfidf-baseline", action="store_true")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the TF-IDF logistic baseline; skips Keras imports and neural training.",
    )
    parser.add_argument(
        "--sweep-json",
        default=None,
        help="JSON file containing a list of run-name and hyperparameter override objects.",
    )
    parser.add_argument("--export-validation-mistakes", action="store_true")
    parser.add_argument("--mistakes-count", type=int, default=20)
    parser.add_argument("--nearest-neighbors", action="store_true")
    parser.add_argument("--neighbors-count", type=int, default=5)
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Import dependencies and exit.",
    )
    parser.add_argument(
        "--allow-missing-deps",
        action="store_true",
        help="Return success when dependency imports fail; useful for repository smoke checks.",
    )
    return parser.parse_args()


def load_dependencies(args: argparse.Namespace) -> dict[str, Any] | None:
    need_keras = not args.baseline_only
    need_sklearn = args.baseline_only or args.run_tfidf_baseline or args.nearest_neighbors
    if need_keras:
        os.environ.setdefault("KERAS_BACKEND", args.backend)
    try:
        import numpy as np
        import pandas as pd
        deps: dict[str, Any] = {"np": np, "pd": pd}
        if need_keras:
            import keras
            from keras import layers

            deps.update({"keras": keras, "layers": layers})
        if need_sklearn:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, log_loss
            from sklearn.neighbors import NearestNeighbors

            deps.update(
                {
                    "TfidfVectorizer": TfidfVectorizer,
                    "LogisticRegression": LogisticRegression,
                    "accuracy_score": accuracy_score,
                    "log_loss": log_loss,
                    "NearestNeighbors": NearestNeighbors,
                }
            )
    except Exception as exc:  # pragma: no cover - depends on local environment
        message = f"Missing or unusable dependency: {exc}"
        if args.allow_missing_deps:
            print(message)
            print("Dependency check skipped because --allow-missing-deps was supplied.")
            return None
        raise RuntimeError(message) from exc
    return deps


def apply_quick_settings(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.epochs = min(args.epochs, 2)
    args.batch_size = min(args.batch_size, 32)
    args.vocab_size = min(args.vocab_size, 2_000)
    args.max_length = min(args.max_length, 80)
    args.embedding_dim = min(args.embedding_dim, 16)
    args.limit_train = args.limit_train or 512
    args.limit_val = args.limit_val or 128


def make_model(args: argparse.Namespace, keras: Any, layers: Any) -> Any:
    inputs = keras.Input(shape=(args.max_length,), dtype="int32", name="token_ids")
    x = layers.Embedding(
        input_dim=args.vocab_size,
        output_dim=args.embedding_dim,
        mask_zero=True,
        name="token_embedding",
    )(inputs)
    x = layers.GlobalAveragePooling1D(name="average_token_vectors")(x)
    if args.dropout > 0:
        x = layers.Dropout(args.dropout, name="pooled_dropout")(x)
    if args.hidden_dim > 0:
        x = layers.Dense(args.hidden_dim, activation="relu", name="hidden")(x)
        if args.dropout > 0:
            x = layers.Dropout(args.dropout, name="hidden_dropout")(x)
    outputs = layers.Dense(1, name="sentiment_logit")(x)
    model = keras.Model(inputs, outputs, name="bag_of_embeddings_sentiment")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(threshold=0.0, name="accuracy")],
    )
    return model


def majority_accuracy(labels: Any, np: Any) -> float:
    labels = np.asarray(labels).astype("int32")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    return max(positives, negatives) / len(labels)


def prepare_split(args: argparse.Namespace, reviews: list[str], labels: Any, np: Any) -> dict[str, Any]:
    train_idx, val_idx = stratified_split(labels, args.validation_fraction, args.seed, np)
    if args.limit_train is not None:
        train_idx = train_idx[: args.limit_train]
    if args.limit_val is not None:
        val_idx = val_idx[: args.limit_val]

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_reviews": [reviews[int(i)] for i in train_idx],
        "val_reviews": [reviews[int(i)] for i in val_idx],
        "y_train": labels[train_idx],
        "y_val": labels[val_idx],
    }


def run_tfidf_logistic_baseline(
    args: argparse.Namespace,
    split: dict[str, Any],
    deps: dict[str, Any],
) -> dict[str, Any]:
    np = deps["np"]
    TfidfVectorizer = deps["TfidfVectorizer"]
    LogisticRegression = deps["LogisticRegression"]
    accuracy_score = deps["accuracy_score"]
    log_loss = deps["log_loss"]

    vectorizer = TfidfVectorizer(
        tokenizer=clean_and_tokenize,
        token_pattern=None,
        lowercase=False,
        max_features=args.vocab_size,
        ngram_range=(1, 2),
        min_df=2,
    )
    classifier = LogisticRegression(max_iter=1000, random_state=args.seed, solver="liblinear")

    start = time.perf_counter()
    x_train = vectorizer.fit_transform(split["train_reviews"])
    classifier.fit(x_train, np.asarray(split["y_train"]).astype("int32"))
    train_seconds = time.perf_counter() - start

    x_val = vectorizer.transform(split["val_reviews"])
    probabilities = classifier.predict_proba(x_val)[:, 1]
    predictions = (probabilities >= 0.5).astype("int32")
    y_val = np.asarray(split["y_val"]).astype("int32")
    validation_log_loss = log_loss(y_val, probabilities, labels=[0, 1])

    return {
        "model": "tfidf_logistic_regression",
        "features": "word_unigram_bigram_tfidf",
        "dataset_source": str(getattr(args, "resolved_data_source", args.data_source)),
        "reviews_train": int(len(split["train_reviews"])),
        "reviews_validation": int(len(split["val_reviews"])),
        "vocab_size_requested": int(args.vocab_size),
        "vocab_size_observed": int(len(vectorizer.vocabulary_)),
        "train_seconds": float(train_seconds),
        "trainable_parameters": int(classifier.coef_.size + classifier.intercept_.size),
        "majority_val_accuracy": float(majority_accuracy(split["y_val"], np)),
        "validation": {
            "accuracy": float(accuracy_score(y_val, predictions)),
            "log_loss": float(validation_log_loss),
        },
    }


def review_stats(review: str, vocab: dict[str, int], max_length: int) -> dict[str, int | bool]:
    tokens = clean_and_tokenize(review)
    return {
        "token_count": len(tokens),
        "used_token_count": min(len(tokens), max_length),
        "oov_count": sum(1 for token in tokens[:max_length] if token not in vocab),
        "truncated": len(tokens) > max_length,
    }


def write_validation_mistakes(
    path: Path,
    val_indices: Any,
    val_reviews: list[str],
    y_val: Any,
    probabilities: Any,
    vocab: dict[str, int],
    max_length: int,
    pd: Any,
    np: Any,
    limit: int,
) -> str:
    labels = np.asarray(y_val).astype("int32")
    probabilities = np.asarray(probabilities).reshape(-1)
    predictions = (probabilities >= 0.5).astype("int32")
    confidence = np.where(predictions == 1, probabilities, 1.0 - probabilities)
    mistake_positions = np.flatnonzero(predictions != labels)
    order = mistake_positions[np.argsort(-confidence[mistake_positions])][:limit]

    rows = []
    for position in order.tolist():
        stats = review_stats(val_reviews[position], vocab, max_length)
        rows.append(
            {
                "original_index": int(val_indices[position]),
                "true_label": int(labels[position]),
                "predicted_label": int(predictions[position]),
                "probability_positive": float(probabilities[position]),
                "confidence": float(confidence[position]),
                **stats,
                "review": val_reviews[position],
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def average_token_embeddings(x: Any, embedding_matrix: Any, np: Any) -> Any:
    vectors = embedding_matrix[x]
    mask = (x != 0).astype("float32")
    denom = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    pooled = (vectors * mask[..., None]).sum(axis=1) / denom
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.maximum(norms, 1e-12)


def write_nearest_neighbor_reviews(
    path: Path,
    x_train: Any,
    x_val: Any,
    train_reviews: list[str],
    val_reviews: list[str],
    y_train: Any,
    y_val: Any,
    val_indices: Any,
    probabilities: Any,
    embedding_matrix: Any,
    pd: Any,
    np: Any,
    NearestNeighbors: Any,
    query_count: int,
    neighbors_count: int,
) -> str:
    y_train = np.asarray(y_train).astype("int32")
    y_val = np.asarray(y_val).astype("int32")
    probabilities = np.asarray(probabilities).reshape(-1)
    predictions = (probabilities >= 0.5).astype("int32")
    confidence = np.where(predictions == 1, probabilities, 1.0 - probabilities)
    mistake_positions = np.flatnonzero(predictions != y_val)
    if len(mistake_positions) == 0:
        mistake_positions = np.arange(len(y_val))
    query_order = mistake_positions[np.argsort(-confidence[mistake_positions])][:query_count]

    train_vectors = average_token_embeddings(x_train, embedding_matrix, np)
    val_vectors = average_token_embeddings(x_val, embedding_matrix, np)
    n_neighbors = min(neighbors_count, len(train_vectors))
    index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    index.fit(train_vectors)
    distances, neighbor_indices = index.kneighbors(val_vectors[query_order])

    rows = []
    for query_rank, query_position in enumerate(query_order.tolist()):
        for neighbor_rank, train_position in enumerate(neighbor_indices[query_rank].tolist(), start=1):
            rows.append(
                {
                    "query_original_index": int(val_indices[query_position]),
                    "query_true_label": int(y_val[query_position]),
                    "query_predicted_label": int(predictions[query_position]),
                    "query_probability_positive": float(probabilities[query_position]),
                    "query_review": val_reviews[query_position],
                    "neighbor_rank": int(neighbor_rank),
                    "neighbor_train_position": int(train_position),
                    "neighbor_label": int(y_train[train_position]),
                    "cosine_similarity": float(1.0 - distances[query_rank][neighbor_rank - 1]),
                    "neighbor_review": train_reviews[train_position],
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_vocabulary(path: Path, vocab: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(vocab.items(), key=lambda item: item[1])
    path.write_text(
        "token\tindex\n" + "\n".join(f"{token}\t{index}" for token, index in rows) + "\n",
        encoding="utf-8",
    )


def json_safe_history(history: Any) -> dict[str, list[float]]:
    return {key: [float(x) for x in values] for key, values in history.history.items()}


def run_neural_experiment(
    args: argparse.Namespace,
    split: dict[str, Any],
    deps: dict[str, Any],
    run_name: str = "bag_embeddings",
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    np = deps["np"]
    pd = deps["pd"]
    keras = deps["keras"]
    layers = deps["layers"]

    keras.backend.clear_session()
    keras.utils.set_random_seed(args.seed)

    vocab = build_vocabulary(split["train_reviews"], args.vocab_size)
    x_train = vectorize_texts(split["train_reviews"], vocab, args.max_length, np)
    x_val = vectorize_texts(split["val_reviews"], vocab, args.max_length, np)

    model = make_model(args, keras, layers)
    callbacks = []
    if args.early_stopping_patience >= 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=args.early_stopping_patience,
                mode="max",
                restore_best_weights=True,
            )
        )

    start = time.perf_counter()
    history = model.fit(
        x_train,
        split["y_train"],
        validation_data=(x_val, split["y_val"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = time.perf_counter() - start
    val_metrics = model.evaluate(x_val, split["y_val"], batch_size=args.batch_size, verbose=0, return_dict=True)
    val_logits = model.predict(x_val, batch_size=args.batch_size, verbose=0).reshape(-1)
    val_probabilities = 1.0 / (1.0 + np.exp(-val_logits))

    summary = {
        "run_name": run_name,
        "model": "bag_of_embeddings",
        "dataset_source": str(getattr(args, "resolved_data_source", args.data_source)),
        "backend": os.environ.get("KERAS_BACKEND", args.backend),
        "python": platform.python_version(),
        "reviews_train": int(len(split["train_reviews"])),
        "reviews_validation": int(len(split["val_reviews"])),
        "vocab_size_requested": int(args.vocab_size),
        "vocab_size_observed": int(len(vocab) + 2),
        "max_length": int(args.max_length),
        "embedding_dim": int(args.embedding_dim),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "train_seconds": float(train_seconds),
        "trainable_parameters": int(model.count_params()),
        "majority_val_accuracy": float(majority_accuracy(split["y_val"], np)),
        "validation": {key: float(value) for key, value in val_metrics.items()},
    }

    if args.make_submission:
        summary["submission_path"] = make_submission(args, model, vocab, pd, np)

    if args.save_artifacts and artifact_dir is not None:
        save_json(artifact_dir / "history.json", json_safe_history(history))
        write_vocabulary(artifact_dir / "vocabulary.tsv", vocab)
        if args.export_validation_mistakes:
            summary["validation_mistakes_path"] = write_validation_mistakes(
                artifact_dir / "validation_mistakes.csv",
                split["val_idx"],
                split["val_reviews"],
                split["y_val"],
                val_probabilities,
                vocab,
                args.max_length,
                pd,
                np,
                args.mistakes_count,
            )
        if args.nearest_neighbors:
            embedding_matrix = model.get_layer("token_embedding").get_weights()[0]
            summary["nearest_neighbors_path"] = write_nearest_neighbor_reviews(
                artifact_dir / "nearest_neighbors.csv",
                x_train,
                x_val,
                split["train_reviews"],
                split["val_reviews"],
                split["y_train"],
                split["y_val"],
                split["val_idx"],
                val_probabilities,
                embedding_matrix,
                pd,
                np,
                deps["NearestNeighbors"],
                args.mistakes_count,
                args.neighbors_count,
            )
        save_json(artifact_dir / "run_summary.json", summary)

    return summary


def load_sweep_configs(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "runs" in data:
        data = data["runs"]
    if not isinstance(data, list):
        raise ValueError("--sweep-json must contain a list, or an object with a 'runs' list.")
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each sweep entry must be a JSON object.")
    return data


def args_with_overrides(args: argparse.Namespace, overrides: dict[str, Any]) -> tuple[str, argparse.Namespace]:
    run_args = argparse.Namespace(**vars(args))
    run_name = str(overrides.get("run_name", "bag_embeddings"))
    for key, value in overrides.items():
        if key == "run_name":
            continue
        attr = key.replace("-", "_")
        if not hasattr(run_args, attr):
            raise ValueError(f"Unknown sweep override: {key}")
        setattr(run_args, attr, value)
    return run_name, run_args


def flatten_result(summary: dict[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in summary.items() if key != "validation" and not isinstance(value, (dict, list))}
    for key, value in summary.get("validation", {}).items():
        row[f"validation_{key}"] = value
    return row


def make_submission(
    args: argparse.Namespace,
    model: Any,
    vocab: dict[str, int],
    pd: Any,
    np: Any,
) -> str:
    test_path = Path(args.data_dir) / "testData.tsv"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found; cannot create submission.")
    frame = pd.read_csv(test_path, sep="\t", quoting=3)
    missing = {"id", "review"} - set(frame.columns)
    if missing:
        raise ValueError(f"{test_path} is missing required columns: {sorted(missing)}")
    x_test = vectorize_texts(frame["review"].astype(str).tolist(), vocab, args.max_length, np)
    logits = model.predict(x_test, batch_size=args.batch_size, verbose=0).reshape(-1)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    if args.submission_mode == "label":
        sentiments = (probabilities >= 0.5).astype("int32")
    else:
        sentiments = probabilities
    output = Path(args.artifact_dir) / "kaggle_submission.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": frame["id"], "sentiment": sentiments}).to_csv(output, index=False)
    return str(output)


def main() -> int:
    args = parse_args()
    apply_quick_settings(args)
    if args.sweep_json and args.make_submission:
        raise ValueError("--make-submission is not supported with --sweep-json; select one final run first.")
    if args.baseline_only:
        args.run_tfidf_baseline = True
    deps = load_dependencies(args)
    if deps is None:
        return 0
    if args.check_deps:
        print("Dependency check passed.")
        return 0

    np = deps["np"]
    pd = deps["pd"]

    if args.synthetic_data:
        reviews, labels = make_synthetic_dataset(np)
        args.resolved_data_source = "synthetic"
    else:
        reviews, labels, resolved_data_source = load_training_texts(Path(args.data_dir), args.data_source, np)
        args.resolved_data_source = resolved_data_source

    split = prepare_split(args, reviews, labels, np)
    artifact_dir = Path(args.artifact_dir)
    output: dict[str, Any] = {
        "python": platform.python_version(),
        "seed": int(args.seed),
        "dataset_source": str(args.resolved_data_source),
        "data_dir": str(args.data_dir),
        "validation_fraction": float(args.validation_fraction),
    }
    if args.run_tfidf_baseline:
        tfidf_summary = run_tfidf_logistic_baseline(args, split, deps)
        output["tfidf_baseline"] = tfidf_summary
        if args.save_artifacts:
            save_json(artifact_dir / "tfidf_baseline_summary.json", tfidf_summary)
    if args.baseline_only:
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0

    if args.sweep_json:
        configs = load_sweep_configs(Path(args.sweep_json))
        summaries = []
        for index, overrides in enumerate(configs, start=1):
            run_name, run_args = args_with_overrides(args, overrides)
            if run_name == "bag_embeddings":
                run_name = f"bag_embeddings_{index:02d}"
            run_artifact_dir = artifact_dir / run_name if args.save_artifacts else None
            summaries.append(run_neural_experiment(run_args, split, deps, run_name, run_artifact_dir))
        output["sweep_results"] = summaries
        if args.save_artifacts:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            save_json(artifact_dir / "sweep_results.json", summaries)
            pd.DataFrame([flatten_result(summary) for summary in summaries]).to_csv(
                artifact_dir / "sweep_results.csv",
                index=False,
            )
    else:
        summary = run_neural_experiment(args, split, deps, "bag_embeddings", artifact_dir if args.save_artifacts else None)
        if "tfidf_baseline" in output:
            summary["tfidf_baseline"] = output["tfidf_baseline"]
        output["neural_result"] = summary
        if args.save_artifacts:
            save_json(artifact_dir / "run_summary.json", summary)

    print(json.dumps(output, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
