# Neural Network Training Revisited Chapter Code

This directory contains the repeatable experiment used to fill the reference
case study in the neural-network-training revisited chapter. The experiment
distills a smaller Transformer text classifier from a larger teacher on AG
News-style topic classification.

Open `transformer_distillation_walkthrough.ipynb` in Jupyter, Colab, or Kaggle
when you want to inspect the committed reference artifacts, dependency checks,
and guarded quick-run path interactively. The notebook intentionally has no
saved outputs.

The main script writes reference artifacts with stable names. The compact
artifacts used by the chapter are mirrored in `reference_artifacts/`:

- `nntrev-reference-distillation-results.csv`
- `nntrev-reference-distillation-results.json`
- `nntrev-reference-result-values.json`
- `nntrev-reference-training-curves.csv`
- `nntrev-reference-latency-table.csv`
- `nntrev-reference-model-footprint.csv`
- `nntrev-reference-confusion-matrix.csv`
- `nntrev-reference-calibration-summary.csv`
- `nntrev-reference-error-examples.csv`
- `nntrev-reference-results-metadata.txt`

## Environment

The real experiment requires PyTorch, Hugging Face Transformers, Hugging Face
Datasets, NumPy, and Matplotlib. A lightweight dependency check is available
for repository checks:

```sh
python3 transformer_distillation_experiment.py --check-deps --allow-missing-deps
```

## Reference CUDA Run

The reference run is intended for comparable CUDA hardware:

```sh
HF_HOME=<hf-cache> \
python3 transformer_distillation_experiment.py \
  --teacher-checkpoint textattack/bert-base-uncased-ag-news \
  --student-checkpoint prajjwal1/bert-mini \
  --cache-dir <hf-cache> \
  --train-examples 4000 \
  --validation-examples 1000 \
  --test-examples 1000 \
  --student-epochs 3 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --mixed-precision bf16 \
  --output-dir runs/reference-nntrev-ag-news
```

If the pretrained task teacher is unavailable, use `--teacher-epochs 1` with a
general BERT-family checkpoint and report that limitation in the metadata.
By default, public error-example CSVs redact raw AG News text. Add
`--include-error-text` only for private analysis or when the dataset terms
permit redistribution.

## Regenerating Reference Plots

After changing plot styling, regenerate or check the reader-facing PNGs from the
committed CSV artifacts without rerunning the full experiment:

```sh
MPLCONFIGDIR=/tmp/mplconfig \
python regenerate_reference_plots.py \
  --plot calibration --check
```

Omit `--check` to overwrite the selected `nntrev-reference-*.png` files.
