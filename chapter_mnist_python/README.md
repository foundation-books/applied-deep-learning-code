# MNIST Python Chapter Code

This directory contains reader-facing companion code for the MNIST Python
chapter.

Poetry uses the shared manifest at `../pyproject.toml` from this chapter
directory. Check it and run the dependency-free shape diagnostic:

```sh
poetry check
poetry run python mnist_shape_check.py
```

For the first training batch, expect a randomly initialized ten-class model to
start near cross-entropy `log(10) ~= 2.30` and chance accuracy near 10%. If the
initial loss or accuracy is very different, check labels, pixel scaling, output
logits, and loss wiring before changing the architecture.

From the repository root, the full code smoke check is available as:

```sh
make mnist-code-check
```

## Notebook Walkthrough

Open `mnist_keras3_walkthrough.ipynb` in Jupyter, Colab, or Kaggle when you
want to run the Keras 3 lesson one cell at a time. The notebook mirrors the
chapter walkthrough and intentionally has no saved outputs.

Use the `.py` files for repeatable command-line runs, backend checks, and smoke
tests.

## Keras 3

Install Keras and at least one backend. Examples:

```sh
poetry install --with keras-tensorflow
poetry install --with keras-torch
poetry install --with keras-jax
```

Run the chapter model from this directory:

```sh
KERAS_BACKEND=tensorflow poetry run python mnist_keras3.py
KERAS_BACKEND=torch poetry run python mnist_keras3.py
KERAS_BACKEND=jax poetry run python mnist_keras3.py
```

Use a short run while checking a new environment:

```sh
KERAS_BACKEND=torch poetry run python mnist_keras3.py --quick
```

## Direct PyTorch

Install PyTorch and torchvision:

```sh
poetry install --with pytorch
```

Run the chapter model from this directory:

```sh
poetry run python mnist_pytorch.py
```

Use a short run while checking a new environment:

```sh
poetry run python mnist_pytorch.py --quick
```
