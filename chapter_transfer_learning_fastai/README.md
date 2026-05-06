# Food-101 Transfer-Learning Companion Code

This directory contains reader-facing chapter scripts for the fast.ai baseline
(`food101_fastai.py`) and the optional Keras 3 translation
(`food101_keras3.py`).

Poetry uses the shared manifest at `../pyproject.toml` from this chapter
directory.

Install or update the fast.ai chapter environment:

```sh
poetry install --with transfer-learning
```

Quick dependency check:

```sh
poetry run python food101_fastai.py --check-deps
```

The dependency check prints package status only; it does not create a run
directory. Running `food101_fastai.py` with no arguments prints usage and exits
without downloading Food-101. Use `--smoke` for a synthetic smoke run or provide
explicit Food-101 training options when the dataset download and training time
are intended.

Optional Keras 3 backend installs:

```sh
poetry install --with transfer-learning,keras-tensorflow
poetry install --with transfer-learning,keras-torch
poetry install --with transfer-learning,keras-jax
```

Quick Keras dependency check. This command returns success when Keras is not
installed, so the default fast.ai code check can remain lightweight:

```sh
poetry run python food101_keras3.py --check-deps --allow-missing-deps
```

## Notebook Walkthrough

Open `food101_fastai_walkthrough.ipynb` in Jupyter, Colab, or Kaggle when you
want to inspect the fast.ai transfer-learning workflow one cell at a time. The
notebook starts in synthetic smoke mode, intentionally has no saved outputs, and
shells out to `food101_fastai.py` for reproducible artifact generation.

Use the `.py` files as the source of truth for repeatable command-line runs,
backend checks, smoke tests, and full Food-101 experiments.

## Reference Artifacts

The compact records cited by the chapter tables are mirrored under
`reference_artifacts/food101-fastai/`. The
`reference_artifacts/food101-fastai/food101-baseline.csv` file is accompanied by
`command.txt`, `versions.json`, `metadata.json`, and `metrics.json` for the
measured ResNet34 Food-101 validation baseline. Larger
raw run outputs, image-level error-analysis artifacts, and plot exports are not
committed.

Smoke test without downloading Food-101 or pretrained weights. Smoke mode
disables pretrained weights by default unless `--smoke-pretrained` is passed.

```sh
poetry run python food101_fastai.py \
  --smoke --architecture resnet18 --freeze-epochs 1 --epochs 0 \
  --image-size 64 --resize-size 72 --batch-size 4 --max-top-losses 4 \
  --output-dir /tmp/adl-food101-fastai-smoke
```

Full Food-101 baseline, to be run only when the dataset download and training
time are intended:

```sh
poetry run python food101_fastai.py \
  --architecture resnet34 --image-size 224 --resize-size 460 \
  --batch-size 64 --freeze-epochs 1 --epochs 5 --base-lr 3e-3 \
  --output-dir runs/resnet34-baseline
```

The official test split is not evaluated unless `--evaluate-test` is passed.
Use that option only once after selecting a model from validation evidence.

Keras 3 smoke test, after installing one Keras backend:

```sh
poetry run python food101_keras3.py --backend tensorflow --quick
```

Keras 3 Food-101 run using an existing Food-101 download. The script looks for a
fast.ai-style cache automatically, but `--data-root` makes the data provenance
explicit:

```sh
poetry run python food101_keras3.py \
  --backend tensorflow --data-root ~/.fastai/data/food-101 \
  --image-size 224 --batch-size 32 --freeze-epochs 1 --epochs 5 \
  --head-lr 3e-4 --fine-tune-lr 1e-5
```

Runs write `command.txt`, `versions.json`, `metadata.json`, `metrics.json`,
`stage_metrics.csv`, `training_log.csv`, `result_summary.csv`, validation
confusion/top-loss artifacts, and plot files when the optional plotting stack is
available. Full run directories under `runs/` are intentionally ignored because
they can contain large exports and image-level evidence.
