# CNN Basics Chapter Code

This directory contains reader-facing companion code for the convolutional
neural network basics chapter.

Poetry uses the shared manifest at `../pyproject.toml` from this chapter
directory. Check it with:

```sh
poetry check
```

From the repository root, the full code smoke check is available as:

```sh
make cnn-code-check
```

## Notebook Walkthrough

Open `cifar10_keras3_walkthrough.ipynb` in Jupyter, Colab, or Kaggle when you
want to run the CNN basics lesson one cell at a time. The notebook mirrors the
chapter walkthrough and intentionally has no saved outputs.

Use `cifar10_keras3.py` for repeatable command-line runs, backend checks,
smoke tests, and figure generation. The notebook's optional plotting cell
requires Matplotlib, available through the shared `figures` dependency group.

## Keras 3 CIFAR-10

Install Keras and at least one backend. Examples:

```sh
poetry install --with keras-tensorflow
poetry install --with keras-torch
poetry install --with keras-jax
```

If `KERAS_BACKEND` is not already set, `cifar10_keras3.py` defaults to
TensorFlow. You can force another backend with `KERAS_BACKEND=...` or
`--backend`.

Run a quick environment check before a full run:

```sh
KERAS_BACKEND=tensorflow poetry run python cifar10_keras3.py --quick
```

Run the same checks without downloading CIFAR-10:

```sh
poetry run python cifar10_keras3.py --quick --synthetic-data --epochs 1
```

Run a full baseline with an explicit backend. This reports validation metrics
only, so it is appropriate while you are still tuning:

```sh
poetry run python cifar10_keras3.py --backend tensorflow
```

After selecting a final model by validation evidence, add `--evaluate-test` to
report the held-out test result once:

```sh
poetry run python cifar10_keras3.py --backend torch --epochs 10 --batch-size 128 --seed 1234 --validation-size 5000 --save-figures --evaluate-test
```

The script also accepts `--backend`, `--epochs`, `--batch-size`, `--seed`,
`--validation-size`, `--data-dir`, `--synthetic-data`, `--figure-dir`,
`--save-figures`, `--save-image-grid`, and `--evaluate-test`. When `--data-dir`
is set, the script sets `KERAS_HOME` before importing Keras so the dataset cache
follows the requested location.
