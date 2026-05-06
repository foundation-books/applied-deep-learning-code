# Vision-Language Models Companion Code

This directory contains the reader-facing companion code for the
vision-language-model chapter. The notebook is the guided walkthrough; the
Python files are the reusable parser, scorer, sample-data generator, and
optional prompt-only generation harness.

Main files:

```text
vlm_reasoning_walkthrough.ipynb
vlm_eval.py
answer_parser.py
sample_visual_math_prompts.jsonl
```

Repository-safe checks:

```sh
poetry install --with vlm
cd chapter_vision_language_models
python -m json.tool vlm_reasoning_walkthrough.ipynb >/dev/null
python vlm_eval.py --unit-test-parser
python vlm_eval.py --check-deps --allow-missing-deps
python vlm_eval.py --write-sample-data data/sample_visual_math
python vlm_eval.py --preview --prompts data/sample_visual_math/prompts.jsonl
```

The bare `python` commands in this README assume the Poetry virtualenv is
active. If it is not active, prefix the `python` invocation with `poetry run`
from the same working directory.

From the repository root, the smoke check is:

```sh
make vlm-code-check
```

## Notebook Walkthrough

Open `vlm_reasoning_walkthrough.ipynb` when you want the chapter homework as an
interactive sequence: define the contract, inspect images, run or paste a
prompt-only baseline, test the parser and reward, compare one adaptation or
fallback ablation, and build the final error taxonomy.

The notebook has no saved outputs. The first code cells can generate a tiny
local PNG dataset under `data/sample_visual_math/`; that directory is ignored by
git.

## Prompt-Only Baseline

A real VLM run requires PyTorch, Transformers, Accelerate, Pillow, and a
compatible checkpoint. Generate the sample images first:

```sh
python vlm_eval.py --write-sample-data data/sample_visual_math
```

Then run a prompt-only baseline in a suitable environment:

```sh
python vlm_eval.py \
  --generate \
  --prompts data/sample_visual_math/prompts.jsonl \
  --model-name Qwen/Qwen2.5-VL-3B-Instruct \
  --max-new-tokens 64 \
  --output runs/vlm_prompt_outputs.jsonl \
  --summary-output runs/vlm_prompt_summary.json \
  --metadata-output runs/vlm_prompt_metadata.json
```

Use `--check-deps` to inspect the installed runtime packages. Use
`--allow-missing-deps` when running repository checks on a machine that is not
intended for VLM inference.

## Data And Scoring

Prompt records are JSONL. Each row should include at least:

```json
{"id": "example_001", "split": "validation", "image_path": "path/to/image.png", "question": "What is shown?", "reference_answer": "42"}
```

Generated-output rows should add `generated_text`. Score an existing JSONL file
with:

```sh
python vlm_eval.py \
  --score-jsonl runs/vlm_prompt_outputs.jsonl \
  --output runs/vlm_scored_outputs.jsonl \
  --summary-output runs/vlm_scored_summary.json
```

For adaptation, keep this scaffold as the scoring harness. Run LoRA SFT,
VLM RL/GRPO, or a controlled fallback ablation in the appropriate GPU notebook,
save outputs with the same fields, and score the held-out split here.
