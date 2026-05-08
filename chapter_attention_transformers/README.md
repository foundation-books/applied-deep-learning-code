# Attention and Transformers Chapter Code

This directory contains reader-facing companion code for the attention and
Transformers chapter. The main script fine-tunes a Hugging Face sequence
classifier on the Jigsaw toxic-comment labels and compares maximum sequence
lengths for the accuracy/speed table in Section 10.

Additional scripts provide the non-Transformer baseline, dataset audit, and
attention-scaling measurements used to interpret that table:

- `attention_transformers_walkthrough.ipynb`: interactive notebook version of
  the chapter's Hugging Face cells, using synthetic data and a tiny checkpoint
  by default for a quick pipeline check.
- `toxic_comments_tfidf_baseline.py`: TF-IDF plus a linear multi-label
  classifier.
- `toxic_comments_data_audit.py`: label counts, split counts, text lengths, and
  tokenizer truncation rates.
- `attention_scaling_benchmark.py`: GRU, 1D CNN, and self-attention timing by
  sequence length.

The Kaggle data is not bundled with the repository. Download the Jigsaw Toxic
Comment Classification Challenge training file and place it at:

```text
chapter_attention_transformers/data/train.csv
```

The CSV must include `comment_text` and the six label columns:
`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

## Environment

Poetry uses the shared manifest at `../pyproject.toml` from this chapter
directory. Install the attention/Transformers dependencies with:

```sh
poetry install --with attention-transformers
```

Run a lightweight dependency check:

```sh
python toxic_comments_transformer.py --check-deps
```

Repository smoke checks use `--allow-missing-deps` so the textbook can still be
checked on machines without a full Transformer training stack.

## Fill The Section 10 Table

Run the controlled max-length comparison:

```sh
python toxic_comments_transformer.py \
  --data-dir data \
  --checkpoint distilbert-base-uncased \
  --max-lengths 128 256 \
  --epochs 3 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --mixed-precision fp16 \
  --save-artifacts
```

Use `--mixed-precision none` on CPU or unsupported GPU hardware. The run writes
`runs/toxic-length-comparison/comparison_table.csv` with the exact fields needed
for the chapter table: setting, maximum length, validation macro F1, and epoch
time. It also writes `run_summary.json`, `versions.json`, `command.txt`, and per
setting false-positive and false-negative example CSV files for error analysis.

For a tiny pipeline check, use synthetic data and a tiny checkpoint:

```sh
python toxic_comments_transformer.py --quick --save-artifacts
```

## Baseline, Audit, And Scaling Runs

Run the non-Transformer baseline on the same CSV:

```sh
python toxic_comments_tfidf_baseline.py \
  --data-dir data \
  --save-artifacts
```

Audit label imbalance and truncation rates:

```sh
python toxic_comments_data_audit.py \
  --data-dir data \
  --max-lengths 128 256 512 \
  --save-artifacts
```

Benchmark sequence-length scaling with PyTorch:

```sh
python attention_scaling_benchmark.py \
  --lengths 64 128 256 512 \
  --save-artifacts
```
