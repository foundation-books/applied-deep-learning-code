# Recurrent Neural Networks Chapter Code

This directory contains reader-facing companion code for the recurrent neural
networks chapter, centered on IMDB sentiment classification with Keras 3. The
homework uses the same shared raw-text IMDB protocol as the embeddings chapter:
place the ACL IMDB folder at `../chapter_embeddings/data/aclImdb/`, or use a
compatible `labeledTrainData.tsv` in that data directory.

Poetry uses the shared manifest at `../pyproject.toml` from this chapter
directory. Check it with:

```sh
poetry check
```

From the repository root, the full code smoke check is available as:

```sh
make rnn-code-check
```

## Notebook Walkthrough

Open `imdb_rnn_keras3_walkthrough.ipynb` in Jupyter, Colab, or Kaggle when you
want to inspect the IMDB recurrent experiment one cell at a time. The notebook
reuses helpers from `imdb_rnn_keras3.py` so the data loading, model definition,
callbacks, metrics, and artifact writing stay aligned with the script.

Use the `.py` file for repeatable command-line runs, backend checks, smoke
tests, and final artifact-producing experiments.

## Keras 3 IMDB RNN

Install Keras and one backend:

```sh
poetry install --with keras-tensorflow
```

Other backend choices are available:

```sh
poetry install --with keras-torch
poetry install --with keras-jax
```

Run a dependency check:

```sh
python imdb_rnn_keras3.py --check-deps
```

Run a quick synthetic-data check without downloading IMDB:

```sh
python imdb_rnn_keras3.py --quick --synthetic-data --model gru
```

Run a validation-only IMDB experiment on the shared raw-text data:

```sh
python imdb_rnn_keras3.py --model lstm --epochs 5 --early-stopping-patience 2 --batch-size 128 --max-length 200 --num-words 10000 --data-source shared-imdb --data-dir ../chapter_embeddings/data --save-artifacts
```

After selecting a final model by validation evidence, add `--evaluate-test` to
evaluate the held-out test set once. This requires a labeled test split, such as
`aclImdb/test`:

```sh
python imdb_rnn_keras3.py --model lstm --epochs 5 --early-stopping-patience 2 --batch-size 128 --max-length 200 --num-words 10000 --data-source shared-imdb --data-dir ../chapter_embeddings/data --save-artifacts --evaluate-test
```

The script also accepts `--model average`, `--model simple-rnn`, `--model gru`,
`--embedding-dim`, `--hidden-size`, `--dropout`, `--recurrent-dropout`,
`--clipnorm`, `--early-stopping-patience`, `--early-stopping-min-delta`,
`--validation-size`, `--limit-train`, `--limit-val`, `--limit-test`, `--seed`,
`--backend`, `--data-source`, `--data-dir`, `--synthetic-data`,
`--artifact-dir`, and `--save-artifacts`. Use `--data-source keras-imdb` only
for legacy comparisons or quick checks that do not need to match the embeddings
chapter split.

Saved artifacts include a CSV training history and a JSON summary with package
versions, platform details, padding ratios, model size, elapsed time, final
validation metrics, and best validation metrics. The script skips test-set
evaluation unless `--evaluate-test` is supplied.
