# Vision Transformers Chapter Code

This directory contains reader-facing companion code for the Vision
Transformers chapter. The notebook is the interactive walkthrough for the
chapter homework; the Python script is the repeatable experiment engine.

Poetry uses the shared manifest at `../pyproject.toml` from this chapter
directory.

Install or update the chapter environment:

```sh
poetry install --with vision-transformers
```

From the repository root, the code smoke check is available as:

```sh
make vit-code-check
```

The smoke check tolerates missing optional ML dependencies so the textbook can
still be checked on machines that are not configured for PyTorch and
Transformers.

## Notebook Walkthrough

Open `pretrained_vit_walkthrough.ipynb` in Jupyter, Colab, or Kaggle when you
want to follow the homework one cell at a time: define the Food-101 split, run a
head-only pretrained baseline, run the LoRA ablation, inspect mistakes, and
reserve the final test measurement until after the recipe is selected. The
notebook intentionally has no saved outputs.

## Repeatable Script

Run a dependency check:

```sh
poetry run python pretrained_vit_experiment.py --check-deps --allow-missing-deps
```

Run the synthetic quick path without downloading data or pretrained weights:

```sh
poetry run python pretrained_vit_experiment.py \
  --quick --output-dir /tmp/adl-vit-smoke --allow-missing-deps
```

Run the Food-101 head-only baseline with a pretrained ViT checkpoint:

```sh
poetry run python pretrained_vit_experiment.py \
  --dataset food101 --download --data-root ~/.cache/torch/datasets \
  --checkpoint google/vit-base-patch16-224-in21k \
  --image-size 224 --resize-size 256 --batch-size 16 \
  --trainable-mode head --epochs 3 \
  --learning-rate 3e-4 --weight-decay 0.01 \
  --output-dir runs/vit-food101-head
```

Run one controlled ablation by changing one main factor: use LoRA adapters on
the ViT attention query and value projections while keeping the same data split,
checkpoint, and image size:

```sh
poetry run python pretrained_vit_experiment.py \
  --dataset food101 --data-root ~/.cache/torch/datasets \
  --checkpoint google/vit-base-patch16-224-in21k \
  --image-size 224 --resize-size 256 --batch-size 16 \
  --trainable-mode lora --epochs 3 \
  --learning-rate 3e-4 --weight-decay 0.01 \
  --lora-r 16 --lora-alpha 16 --lora-dropout 0.1 \
  --lora-target-modules query value \
  --output-dir runs/vit-food101-lora
```

Use `--evaluate-test` only after selecting the final recipe from validation
evidence. If hardware is limited, use `--limit-train` and `--limit-val` for a
controlled Food-101 subset, or switch to `--dataset cifar10` as a smaller
fallback and state the limitation in the report.

After the baseline and LoRA ablation runs finish, write local summary artifacts:

```sh
poetry run python summarize_vit_runs.py \
  --run baseline=runs/vit-food101-head \
  --run ablation=runs/vit-food101-lora \
  --output-dir runs/vit-food101-summary \
  --prefix vit-food101-lora-reference
```

Runs write `metadata.json`, `metrics.json`, `history.csv`,
`result_summary.csv`, `command.txt`, validation confusion/per-class CSV files,
and a validation mistake table. Pass `--save-mistake-gallery` to also save a
PNG grid of validation mistakes when matplotlib is installed.
