# LoRA and QLoRA Adaptation Companion Code

This directory contains the reader-facing scaffold for the LoRA/QLoRA
fine-tuning homework in the "Fine-Tuning Revisited: Efficient LLM Adaptation
with LoRA and QLoRA" chapter.

The main files are:

```text
lora_unsloth_course_assistant.py
lora_unsloth_course_assistant_walkthrough.ipynb
```

Repository checks can run:

```sh
python lora_unsloth_course_assistant.py --check-deps --allow-missing-deps
python -m json.tool lora_unsloth_course_assistant_walkthrough.ipynb >/dev/null
```

From the companion-code root, `make lora-code-check` also parses the script
without writing `__pycache__` bytecode.

A real training run requires a machine and Python environment compatible with
Unsloth, PyTorch, Transformers, TRL, Datasets, and the selected model. Verify the
current Unsloth installation instructions before running the full homework,
because supported models and hardware recommendations change.

The shared Poetry environment installs the standard PyTorch, Transformers, TRL,
Datasets, PEFT, and Accelerate pieces with:

```sh
cd ..
poetry install --with lora
```

Install Unsloth itself from the current upstream instructions for your GPU
runtime before running the full training commands.

The homework workflow is:

1. Prepare a JSONL file with `question` and `answer` fields.
2. Reserve held-out validation prompts before training.
3. Run a prompt-only baseline with fixed decoding settings.
4. Run LoRA or QLoRA supervised fine-tuning.
5. Inspect a tokenized batch and reload the saved adapter for held-out
   generation.
6. Run one ablation by changing one setting, such as LoRA rank or sequence
   length.
7. Compare validation quality, training time, memory, and trainable parameters.

## Notebook Walkthrough

Open `lora_unsloth_course_assistant_walkthrough.ipynb` in Jupyter, Colab, or
Kaggle when you want to run the LoRA/QLoRA homework one cell at a time. The
notebook mirrors the chapter's homework cells and intentionally has no saved
outputs.

Use the `.py` file for repeatable command-line runs, metadata generation,
dependency checks, and repository smoke checks.

Example baseline and training commands, to run only inside a compatible Unsloth
GPU environment:

```sh
python lora_unsloth_course_assistant.py --baseline --validation-data validation.jsonl
python lora_unsloth_course_assistant.py --inspect-tokenized-batch --data train.jsonl
python lora_unsloth_course_assistant.py --train --data train.jsonl --run-kind lora
python lora_unsloth_course_assistant.py --evaluate-adapter --validation-data validation.jsonl --adapter-dir runs/unsloth_lora/lora_adapter
python lora_unsloth_course_assistant.py --train --data train.jsonl --run-kind ablation --run-name rank_8 --lora-rank 8
```

The baseline command writes generated validation answers and metadata. Training
commands write a LoRA adapter plus metadata containing package versions,
base-model provenance, trainable parameter count, runtime, and available CUDA
memory information. Use `--model-revision`, `--model-license`, and
`--model-card-url` when preparing an adapter that may be shared outside a local
homework run. The adapter-evaluation command proves that the saved adapter can
be loaded from disk and used on the same held-out prompts.

Before publishing an adapter, review the base model license and model card, the
training data terms, and the generated metadata. The MIT license for this
companion code does not relicense the base model, adapter weights, or training
data.
