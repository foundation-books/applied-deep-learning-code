# CNNs Revisited Companion Code

This directory contains the reader-facing PyTorch script for the ConvNeXt
Food-101 homework.

Poetry uses the shared manifest at `../pyproject.toml` from this chapter
directory.

Install or update the environment:

```sh
poetry install --with transfer-learning
```

## Notebook Walkthrough

Open `convnext_food101_walkthrough.ipynb` in Jupyter, Colab, or Kaggle when you
want to inspect the ConvNeXt Food-101 workflow one cell at a time. The notebook
starts in synthetic smoke mode, intentionally has no saved outputs, and shells
out to `convnext_food101_pytorch.py` for reproducible artifact generation.

Use `convnext_food101_pytorch.py` as the source of truth for repeatable
command-line runs, dependency checks, smoke tests, validation artifacts, and
full Food-101 experiments.

Dependency check:

```sh
poetry run python convnext_food101_pytorch.py --check-deps --allow-missing-deps
```

Synthetic smoke test without downloading Food-101 or pretrained weights:

```sh
poetry run python convnext_food101_pytorch.py \
  --quick --output-dir /tmp/adl-convnext-food101-smoke --allow-missing-deps
```

Full Food-101 run using an existing Food-101 download:

```sh
poetry run python convnext_food101_pytorch.py \
  --data-root ~/.fastai/data/food-101 \
  --model-variant convnext_tiny \
  --image-size 224 --resize-size 256 --batch-size 32 \
  --freeze-epochs 1 --epochs 4 \
  --head-lr 3e-4 --fine-tune-lr 1e-5 --weight-decay 0.05 \
  --save-plots --output-dir runs/convnext-tiny-food101
```

If Food-101 is not already available, let TorchVision download it first:

```sh
poetry run python convnext_food101_pytorch.py \
  --download --download-root ~/.cache/torch/datasets \
  --model-variant convnext_tiny \
  --limit-train 512 --limit-val 128 \
  --freeze-epochs 1 --epochs 0 \
  --save-plots --output-dir runs/convnext-tiny-food101-debug
```

Use `--evaluate-test` only after selecting the final model from validation
evidence. Runs write `metadata.json`, `metrics.json`, `history.csv`,
`command.txt`, `best_model.pt`, validation confusion and per-class CSV files,
validation prediction examples, optional PNG plots, and, when requested,
matching test artifacts.

The reference run reported in the chapter used this command shape with
`--evaluate-test` after the validation recipe was fixed. It ran on a CUDA
reference workstation with PyTorch 2.11.0+cu130, TorchVision 0.26.0+cu130, CUDA,
and mixed precision.
The reader-facing compact reference artifacts are mirrored under
`reference_artifacts/food101-convnext-reference/`. They record 78.36%
validation top-1 accuracy, 83.52% final test top-1 accuracy, 96.61% final test
top-5 accuracy, and 1234.1 seconds elapsed for ConvNeXt-Tiny on Food-101.
