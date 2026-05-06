# Neural Network Training Chapter Code

This directory contains a guided notebook for the neural-network training
chapter. The notebook is the reader-facing lab surface; the existing PyTorch
runner in `../chapter_cnn_architectures/resnet50_cifar10_pytorch.py` remains the
canonical implementation for model construction, training, evaluation, and
artifact writing.

Install the PyTorch and figure dependencies from the shared repository environment:

```sh
cd ..
poetry install --with pytorch,figures
```

Then open `cifar10_training_lab.ipynb` with a Python kernel that uses the same
environment. The notebook starts with a synthetic-data smoke run, then provides
bounded CIFAR-10 cells for a baseline, one optimizer or schedule change, one
regularization change, and a guarded final selected test run.

Notebook artifacts are written under `runs/`, which is ignored by Git. Add
`--evaluate-test` only for the final selected model, matching the chapter's
validation-first protocol.

The measured artifact table in the chapter has reader-facing compact records
under `reference_artifacts/table5-4/`. That directory mirrors the bounded
ResNet18 history CSVs, training-curve PNGs, confusion matrices, per-class
accuracy CSVs, and metadata text files used for the table.

Check the notebook structure without running the training cells:

```sh
python -m json.tool chapter_neural_network_training/cifar10_training_lab.ipynb >/tmp/adl-cifar10-training-lab.json
```
