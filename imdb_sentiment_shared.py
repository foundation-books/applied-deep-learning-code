#!/usr/bin/env python3
"""Shared raw-text IMDB sentiment helpers for embeddings and RNN chapters."""

from __future__ import annotations

from collections import Counter
import csv
import html
from pathlib import Path
import re
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9']+")


def clean_and_tokenize(text: str) -> list[str]:
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return TOKEN_RE.findall(text.lower())


def build_vocabulary(texts: list[str], vocab_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(clean_and_tokenize(text))

    # 0 is reserved for padding and 1 for out-of-vocabulary tokens.
    most_common = counts.most_common(max(vocab_size - 2, 0))
    return {token: index + 2 for index, (token, _) in enumerate(most_common)}


def vectorize_texts(texts: list[str], vocab: dict[str, int], max_length: int, np: Any) -> Any:
    x = np.zeros((len(texts), max_length), dtype="int32")
    for row, text in enumerate(texts):
        ids = [vocab.get(token, 1) for token in clean_and_tokenize(text)]
        ids = ids[:max_length]
        if ids:
            x[row, : len(ids)] = ids
    return x


def mean_token_length(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return float(sum(len(clean_and_tokenize(text)) for text in texts) / len(texts))


def make_synthetic_dataset(np: Any, repeats: int = 120) -> tuple[list[str], Any]:
    positive = [
        "a warm funny movie with excellent acting",
        "beautiful story and great performances",
        "smart enjoyable film with a satisfying ending",
        "wonderful direction and memorable characters",
    ]
    negative = [
        "a dull boring movie with weak acting",
        "terrible story and awful performances",
        "confusing unpleasant film with a disappointing ending",
        "poor direction and forgettable characters",
    ]
    reviews: list[str] = []
    labels: list[int] = []
    for _ in range(repeats):
        for text in positive:
            reviews.append(text)
            labels.append(1)
        for text in negative:
            reviews.append(text)
            labels.append(0)
    return reviews, np.asarray(labels, dtype="float32")


def stratified_split(labels: Any, validation_fraction: float, seed: int, np: Any) -> tuple[Any, Any]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    labels = np.asarray(labels)
    validation_size = int(round(len(labels) * validation_fraction))
    return stratified_split_count(labels, validation_size, seed, np)


def stratified_split_count(labels: Any, validation_size: int, seed: int, np: Any) -> tuple[Any, Any]:
    labels = np.asarray(labels)
    if validation_size <= 0 or validation_size >= len(labels):
        raise ValueError("validation_size must be positive and smaller than the training set.")

    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        label_fraction = len(indices) / len(labels)
        val_count = max(1, int(round(validation_size * label_fraction)))
        val_count = min(val_count, len(indices) - 1)
        val_parts.append(indices[:val_count])
        train_parts.append(indices[val_count:])

    train_indices = np.concatenate(train_parts)
    val_indices = np.concatenate(val_parts)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def read_labeled_tsv(path: Path, np: Any) -> tuple[list[str], Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    reviews: list[str] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter="\t", quoting=csv.QUOTE_NONE)
        fieldnames = set(reader.fieldnames or [])
        missing = {"review", "sentiment"} - fieldnames
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            reviews.append(str(row["review"]))
            labels.append(int(str(row["sentiment"]).strip().strip('"')))
    return reviews, np.asarray(labels, dtype="float32")


def find_acl_imdb_root(data_dir: Path) -> Path | None:
    candidates = [data_dir, data_dir / "aclImdb"]
    for candidate in candidates:
        if (candidate / "train" / "pos").is_dir() and (candidate / "train" / "neg").is_dir():
            return candidate
    return None


def load_acl_imdb_split(data_dir: Path, split: str, np: Any) -> tuple[list[str], Any]:
    root = find_acl_imdb_root(data_dir)
    if root is None:
        raise FileNotFoundError(
            f"Could not find aclImdb under {data_dir}. Expected aclImdb/{split}/pos and aclImdb/{split}/neg."
        )

    reviews: list[str] = []
    labels: list[int] = []
    for label_name, label in (("neg", 0), ("pos", 1)):
        split_dir = root / split / label_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"{split_dir} not found.")
        for path in sorted(split_dir.glob("*.txt")):
            reviews.append(path.read_text(encoding="utf-8", errors="replace"))
            labels.append(label)
    if not reviews:
        raise ValueError(f"No IMDB review files found under {root / split}.")
    return reviews, np.asarray(labels, dtype="float32")


def load_training_texts(data_dir: Path, data_source: str, np: Any) -> tuple[list[str], Any, str]:
    train_tsv = data_dir / "labeledTrainData.tsv"
    if data_source in {"auto", "shared-imdb", "kaggle-tsv"} and train_tsv.exists():
        reviews, labels = read_labeled_tsv(train_tsv, np)
        resolved = "shared-imdb-tsv" if data_source != "kaggle-tsv" else "kaggle-tsv"
        return reviews, labels, resolved

    if data_source in {"auto", "shared-imdb", "acl-imdb"}:
        root = find_acl_imdb_root(data_dir)
        if root is not None:
            reviews, labels = load_acl_imdb_split(root, "train", np)
            return reviews, labels, "acl-imdb-train"

    if data_source == "kaggle-tsv":
        raise FileNotFoundError(f"{train_tsv} not found. Download the Kaggle TSV or use --data-source acl-imdb.")
    raise FileNotFoundError(
        f"No shared IMDB data found under {data_dir}. Expected labeledTrainData.tsv or aclImdb/."
    )


def load_optional_labeled_test_texts(data_dir: Path, data_source: str, np: Any) -> tuple[list[str], Any, str] | None:
    if data_source in {"auto", "shared-imdb", "acl-imdb"}:
        root = find_acl_imdb_root(data_dir)
        if root is not None and (root / "test" / "pos").is_dir() and (root / "test" / "neg").is_dir():
            reviews, labels = load_acl_imdb_split(root, "test", np)
            return reviews, labels, "acl-imdb-test"

    for filename in ("labeledTestData.tsv", "testDataLabeled.tsv", "test_labeled.tsv"):
        path = data_dir / filename
        if path.exists():
            reviews, labels = read_labeled_tsv(path, np)
            return reviews, labels, filename
    return None
