# Image Generation Chapter Code

This directory contains the repeatable experiment engine for the image
generation chapter. The script is self-contained: it creates a small synthetic
`32 x 32` image dataset with permitted provenance, then runs compact
autoencoder, VAE, DCGAN, diffusion noising/denoising, and LoRA-style
diffusion fine-tuning experiments.

The generated data are intentionally simple. They are evidence for workflow,
reporting, speed, memory, ablation, and failure-case discipline; they are not a
claim that a tiny classroom model matches large public text-to-image systems.

Open `image_generation_walkthrough.ipynb` in Jupyter, Colab, or Kaggle when you
want to inspect the reference artifacts and run the quick synthetic experiment
one cell at a time. The notebook intentionally has no saved outputs.

Run the dependency check:

```sh
python image_generation_experiments.py --check-deps --allow-missing-deps
```

Run a short smoke test:

```sh
python image_generation_experiments.py \
  --mode quick \
  --output-dir /tmp/adl-image-generation-smoke
```

Run the reference experiment on a GPU machine:

```sh
python image_generation_experiments.py \
  --mode reference \
  --device auto \
  --output-dir runs/image-generation-reference
```

The run writes PNG grids, CSV tables, a metadata text file, and a
`thor1-imagegen-result-values.json` file. The compact JSON and CSV records cited
by the chapter tables are mirrored in `reference_artifacts/`:
`thor1-imagegen-result-values.json`, `thor1-imagegen-ablation-table.csv`, and
`thor1-imagegen-runtime-memory-table.csv`. Figure-only PNG grids remain rendered
in the book PDF.
