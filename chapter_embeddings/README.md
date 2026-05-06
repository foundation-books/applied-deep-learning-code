# Embeddings Chapter Companion Code

This directory contains reader-facing code connected to the embeddings and
learned representations chapter.

The chapter homework uses the shared IMDB sentiment protocol also used by the
RNN chapter. The preferred source is the public ACL IMDB v1 raw-text dataset
from Maas et al. The textbook source does not bundle the reviews; download the
dataset and place the extracted folder at:

```text
chapter_embeddings/data/aclImdb/
```

The script also accepts a converted three-column TSV named
`labeledTrainData.tsv` with `id`, `sentiment`, and `review` columns. Kaggle's
**Bag of Words Meets Bags of Popcorn** TSV files have that shape and remain
compatible for optional submission work, but the chapter comparison should use
the same source and split as the RNN homework.

The required homework model is:

```text
tokenizer
  -> Embedding
  -> GlobalAveragePooling1D
  -> Dense sentiment classifier
```

## Environment

Poetry uses the shared manifest at `../pyproject.toml` from this chapter
directory. The script uses pandas, scikit-learn for the TF-IDF baseline and
nearest-neighbor inspection, and Keras 3 for the neural model. Choose the
embeddings group and one backend group:

```sh
poetry install --with embeddings,keras-tensorflow
poetry install --with embeddings,keras-torch
poetry install --with embeddings,keras-jax
```

Run a dependency check:

```sh
poetry run python kaggle_bag_of_embeddings_sentiment.py --check-deps
```

Repository smoke checks use `--allow-missing-deps` so the textbook can be
checked on machines that do not have a full deep-learning stack installed.

## Notebook Walkthrough

Open `kaggle_bag_of_embeddings_sentiment_walkthrough.ipynb` in Jupyter, Colab,
or Kaggle when you want to work through the homework one cell at a time. The
notebook previews tokenization and vectorization, runs the same companion
script, and inspects the saved result artifacts for the homework table and
error analysis. It intentionally has no saved outputs. The historical filename
is kept so older links still work.

Use `kaggle_bag_of_embeddings_sentiment.py` for repeatable command-line runs,
backend checks, sweeps, optional Kaggle submissions, and repository smoke
checks.

## Train The Homework Model

Run the non-neural TF-IDF logistic regression baseline:

```sh
poetry run python kaggle_bag_of_embeddings_sentiment.py \
  --data-dir data \
  --baseline-only \
  --save-artifacts
```

Train the bag-of-embeddings model and record validation examples for error
analysis:

```sh
poetry run python kaggle_bag_of_embeddings_sentiment.py \
  --backend tensorflow \
  --data-dir data \
  --epochs 8 \
  --batch-size 128 \
  --vocab-size 20000 \
  --max-length 400 \
  --embedding-dim 64 \
  --run-tfidf-baseline \
  --export-validation-mistakes \
  --nearest-neighbors \
  --save-artifacts
```

Create a quick local smoke run without real IMDB data:

```sh
poetry run python kaggle_bag_of_embeddings_sentiment.py \
  --synthetic-data \
  --quick \
  --backend tensorflow
```

Run a controlled sweep from a JSON file:

```sh
poetry run python kaggle_bag_of_embeddings_sentiment.py \
  --backend tensorflow \
  --data-dir data \
  --sweep-json sweep.json \
  --run-tfidf-baseline \
  --save-artifacts
```

The sweep file can be either a list of run objects or an object with a `runs`
list:

```json
[
  {"run_name": "d64_len400", "embedding_dim": 64, "max_length": 400},
  {"run_name": "d128_len400", "embedding_dim": 128, "max_length": 400},
  {"run_name": "d64_len200", "embedding_dim": 64, "max_length": 200}
]
```

Create a Kaggle submission after selecting a model by validation evidence:

```sh
poetry run python kaggle_bag_of_embeddings_sentiment.py \
  --data-dir data \
  --make-submission \
  --save-artifacts
```

The script prints a JSON summary with validation metrics, training time,
trainable parameter count, and the majority-class validation baseline. With
`--save-artifacts`, it writes `run_summary.json`, `history.json`,
`vocabulary.tsv`, optional `tfidf_baseline_summary.json`, optional
`validation_mistakes.csv`, optional `nearest_neighbors.csv`, and sweep-level
`sweep_results.json`/`sweep_results.csv`. Use those measured values to replace
the `TBD` cells in the homework table. Tune only on the validation split, record
training time and parameter count, and use any labeled test split or Kaggle
submission only after final model selection.
