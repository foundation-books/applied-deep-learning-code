# From Transformer to LLMs Companion Code

This directory contains the reader-facing prompt-only scaffold for the "From
Transformer to Large Language Models" chapter. It is for running and measuring a
pretrained instruction model, not for fine-tuning it.

The main script is:

```text
llm_prompt_benchmark.py
```

The interactive walkthrough is:

```text
llm_prompt_benchmark_walkthrough.ipynb
```

Open the notebook from a repository checkout in Jupyter, Colab, or Kaggle when
you want to follow the homework one step at a time: write validation prompts,
preview formatting, count tokens, run the greedy baseline, run one decoding
ablation, and prepare the scoring table. The notebook intentionally has no saved
outputs and calls `llm_prompt_benchmark.py` instead of duplicating the benchmark
implementation.

Use the `.py` script for repeatable command-line runs, dependency checks, and
repository smoke tests.

The script is intentionally conservative. Repository checks can run:

```sh
python llm_prompt_benchmark.py --check-deps --allow-missing-deps
```

From the companion-code root, `make llm-code-check` also parses the script
without writing `__pycache__` bytecode and previews the committed sample prompt
file.

A real generation run requires a machine and Python environment compatible with
PyTorch, Transformers, and the selected checkpoint. Start with a small instruct
model that fits the available hardware.

The homework workflow is:

1. Prepare held-out prompts with `question` and optional `reference_answer`
   fields.
2. Inspect the prompt text before loading a model.
3. Count tokens with the same tokenizer used by the checkpoint. For question
   records, the scaffold uses the tokenizer chat template by default when one is
   available.
4. Run prompt-only generation with fixed decoding settings.
5. Run one decoding ablation, such as greedy decoding versus top-p sampling.
6. Compare answer quality, token counts, first-token latency, tokens per second,
   and memory metadata.
7. After manually scoring the generated answers, aggregate the rubric scores into
   a JSON artifact beside the generations and metadata.

Example commands:

```sh
python llm_prompt_benchmark.py --preview --prompts sample_prompts.jsonl
python llm_prompt_benchmark.py --estimate-tokens --prompts sample_prompts.jsonl
python llm_prompt_benchmark.py \
  --generate \
  --prompts sample_prompts.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --max-new-tokens 128 \
  --use-chat-template \
  --temperature 0

python llm_prompt_benchmark.py \
  --aggregate-scores \
  --scores-csv runs/greedy_scores.csv \
  --output runs/greedy_outputs.jsonl \
  --metadata-output runs/greedy_metadata.json \
  --score-output runs/greedy_score_summary.json
```

Example JSONL record:

```json
{"question": "How do I improve validation accuracy in an image classifier?", "reference_answer": "Start by checking the validation protocol and data split. Then compare one controlled change at a time, such as augmentation, learning rate, model size, or input resolution. Do not tune on the final test set."}
```

Records with a raw `prompt` field are treated as already formatted. Use
`--no-use-chat-template` only when intentionally comparing against a plain
`Question:`/`Answer:` prompt or when the selected tokenizer has no chat template.

The script includes a tiny built-in prompt set for smoke tests and demonstration.
It is not a substitute for a held-out validation set.
