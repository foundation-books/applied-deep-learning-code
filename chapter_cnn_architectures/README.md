# CNN Architectures Chapter Code

This directory contains reader-facing companion code for the CNN architectures
chapter, centered on controlled CIFAR-10 experiments with ResNet-family models.
`resnet18` is the default student-scale variant; `resnet34`, `resnet50`, and
`tiny-resnet` are also available through `--model-variant`. The script filenames
retain `resnet50` for compatibility with the committed verification artifacts;
the `--model-variant` argument controls the actual model.

The notebook `resnet_cifar10_walkthrough.ipynb` is the student-facing walkthrough.
Use it to inspect ResNet variants, compare CIFAR-style and ImageNet-style stems,
trace tensor shapes, count parameters, and run the quick synthetic-data
verification path. Open it from the repository root or from this directory with
a kernel that has the PyTorch companion-code dependencies installed. The Python
scripts remain the source of truth for repeatable training runs, saved figures,
metadata, and smoke checks.

Poetry uses the shared manifest at `../pyproject.toml` from this chapter
directory. Check it with:

```sh
poetry check
```

From the repository root, the full code smoke check is available as:

```sh
make cnn-arch-code-check
```

## Reference Artifacts

The compact records cited by the chapter tables are mirrored in
`reference_artifacts/`:

- `reference_artifacts/cnn-architecture-quantitative-comparison.csv` records the
  representative architecture scale table and source notes.
- `resnet50-cifar10-pytorch/` contains the bounded ResNet50 CIFAR-10
  verification history, training-curve PNG, confusion matrix, per-class
  accuracy table, and metadata.

## Keras 3 ResNet

Install Keras and one backend:

```sh
poetry install --with keras-tensorflow,figures
```

Other backend choices are available:

```sh
poetry install --with keras-torch,figures
poetry install --with keras-jax,figures
```

Run a dependency check:

```sh
poetry run python resnet50_cifar10_keras3.py --check-deps
```

Run a quick synthetic-data check:

```sh
poetry run python resnet50_cifar10_keras3.py --quick --synthetic-data
```

Run a student-scale full CIFAR-10 experiment with the CIFAR-style stem:

```sh
poetry run python resnet50_cifar10_keras3.py --backend tensorflow --epochs 50 --batch-size 128 --stem cifar --model-variant resnet18 --augmentation basic --schedule cosine --save-figures
```

## Direct PyTorch ResNet

Install the direct PyTorch stack:

```sh
poetry install --with pytorch,figures
```

Run a dependency check:

```sh
poetry run python resnet50_cifar10_pytorch.py --check-deps
```

Run a quick synthetic-data check:

```sh
poetry run python resnet50_cifar10_pytorch.py --quick --synthetic-data
```

Run the bounded real-data verification command used for the committed PyTorch
artifact format:

```sh
poetry run python resnet50_cifar10_pytorch.py --epochs 1 --batch-size 32 --stem cifar --model-variant resnet50 --optimizer adamw --learning-rate 0.001 --weight-decay 0.0005 --schedule constant --augmentation basic --validation-size 5000 --limit-train 512 --limit-val 256 --limit-test 256 --evaluate-test --save-figures
```

Run a student-scale full CIFAR-10 experiment:

```sh
poetry run python resnet50_cifar10_pytorch.py --epochs 50 --batch-size 128 --stem cifar --model-variant resnet18 --augmentation basic --schedule cosine --save-figures
```

Add `--evaluate-test` only for the final selected or verification run. Ordinary
model-selection runs should rely on validation metrics and keep the held-out
test set locked.
