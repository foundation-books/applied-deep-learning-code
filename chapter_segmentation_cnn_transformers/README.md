# Segmentation Companion Workflow

This directory contains the reader-facing code for the segmentation chapter.
The dependency-light workflow checks masks, computes toy metrics, writes dataset
templates, records environment information, and summarizes result files. The
optional training script fine-tunes SegFormer on a CSV of medical image-mask
pairs when PyTorch, Transformers, Pillow, and NumPy are available.

Open `segmentation_medical_walkthrough.ipynb` in Jupyter, Colab, or Kaggle when
you want to run the dependency-light workflow one cell at a time. The notebook
intentionally has no saved outputs and keeps real training behind an explicit
flag.

Create a dataset template:

```sh
python segmentation_medical_workflow.py make-template --output-dir /tmp/seg-template
```

Compute the chapter's small IoU and Dice example:

```sh
python segmentation_medical_workflow.py metric-example --output-dir /tmp/seg-metrics
```

Inspect text masks listed in a CSV file with columns `image_path` and
`mask_path`:

```sh
python segmentation_medical_workflow.py inspect-masks \
  --pairs-csv /path/to/pairs.csv \
  --output-dir /tmp/seg-mask-audit
```

Write a dry-run command for a future SegFormer fine-tuning run:

```sh
python segmentation_medical_workflow.py train-command \
  --data-root /path/to/kvasir-seg \
  --model nvidia/segformer-b0-finetuned-ade-512-512 \
  --image-size 256 \
  --epochs 10 \
  --output-dir runs/kvasir-segformer-b0 \
  --dry-run
```

Check optional training dependencies:

```sh
python segformer_medical_finetune.py dependency-check --allow-missing-deps
```

Fine-tune SegFormer-B0 after preparing `pairs.csv` with `image_path`,
`mask_path`, and `split` columns:

```sh
python segformer_medical_finetune.py train \
  --pairs-csv /path/to/kvasir-seg/pairs.csv \
  --data-root /path/to/kvasir-seg \
  --output-dir runs/kvasir-segformer-b0-256 \
  --model-name nvidia/segformer-b0-finetuned-ade-512-512 \
  --image-size 256 \
  --epochs 5 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --thresholds 0.3 0.5 0.7 \
  --amp
```

The training script writes `config.json`, `dataset_provenance.json`,
`environment.json`, `history.csv`, `validation_metrics.csv`, `metrics.json`,
optional `test_metrics.csv`, optional `training_curve.png`,
`qualitative_grid.png`, and `failure_grid.png`. Validation selects the
checkpoint and threshold; test metrics are written only when the CSV contains a
`test` split.

The default metadata fields are written for Kvasir-SEG. Kvasir-SEG is restricted
to research and educational use, requires citation of the dataset paper, and
does not permit commercial use without prior written permission. Do not publish
qualitative grids, failure grids, copied images, or derived masks unless that
release satisfies the upstream dataset terms.

After experiments have written metrics JSON files, summarize them:

```sh
python segmentation_medical_workflow.py summarize-runs \
  --runs-dir runs/segmentation \
  --output-dir runs/segmentation-summary
```

Run the dependency-light smoke check from the repository root:

```sh
make segmentation-code-check
```

Real training is intentionally not launched by the smoke check because it needs
the dataset and optional ML packages.
