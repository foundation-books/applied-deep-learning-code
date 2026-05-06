# Reinforcement Learning for LLM Training Companion Code

This directory contains a small scaffold for the chapter homework. It focuses on
reward-function testing, sample-data format, dependency checks, and a narrow
optional Unsloth GRPO/GSPO arithmetic run. If a versioned Unsloth notebook is a
better fit for the available GPU environment, reuse the reward tests,
metadata template, and held-out evaluation discipline here rather than relying on
the notebook defaults alone.

Main file:

```text
rl_lora_unsloth_reasoning.py
```

The reward helper is in `reward_functions.py`, and the small example dataset is
in `sample_reasoning_prompts.jsonl`.

The reader-facing notebook is:

```text
rl_lora_unsloth_reasoning_walkthrough.ipynb
```

Use the notebook in Jupyter, Colab, or Kaggle when you want to run the homework
workflow one cell at a time. It intentionally delegates reward tests, sample
data, metadata, and optional GRPO/GSPO runs to the `.py` files so the scripts
remain the repeatable source of truth.

Install the standard shared runtime pieces with:

```sh
cd ..
poetry install --with rl-llm
```

Install Unsloth itself from the current upstream instructions for your GPU
runtime before running the optional GRPO/GSPO trainer. Keep that provider
installation outside the locked smoke-test environment, and record the exact
install command, package versions, model revision, and hardware with any run
artifacts you compare or publish.

The optional Unsloth GRPO/GSPO experiment runner is:

```text
unsloth_grpo_gspo_reasoning_experiment.py
```

Repository-safe checks:

```sh
python -m json.tool rl_lora_unsloth_reasoning_walkthrough.ipynb >/dev/null
python rl_lora_unsloth_reasoning.py --check-deps --allow-missing-deps
python rl_lora_unsloth_reasoning.py --unit-test-rewards
python unsloth_grpo_gspo_reasoning_experiment.py --check-deps --allow-missing-deps
python rl_lora_unsloth_reasoning.py --write-sample-data data
python rl_lora_unsloth_reasoning.py --write-metadata-template data/run_metadata_template.json
```

From the companion-code root, `make rl-llm-code-check` also parses the chapter
scripts without writing `__pycache__` bytecode.

The sample JSONL records contain `question`, `answer`, and `final_answer`
fields. They are examples for checking the pipeline, not enough data for a
meaningful model-quality claim.

The reward unit tests intentionally include parser-exploit cases such as
multiple answer tags, prose around the tag, empty tags, and contradictory text.
Correctness reward is given only when the completion is exactly one valid
`<answer>...</answer>` tag with the expected payload.

The expected homework workflow is:

1. Prepare training and held-out validation prompts.
2. Unit-test the reward function.
3. Run a prompt-only baseline.
4. Run one Unsloth LoRA or QLoRA GRPO/GSPO job.
5. Run one ablation by changing exactly one setting.
6. Compare held-out score, invalid-output rate, length, runtime, throughput, and
   peak GPU memory.
7. Record model name and revision, seed, package versions, hardware, LoRA rank
   and alpha, batch settings, reward weights, and artifact paths.

## Optional Unsloth GRPO/GSPO Run

The `unsloth_grpo_gspo_reasoning_experiment.py` runner uses Unsloth plus TRL's
`GRPOTrainer` for a small arithmetic task. Use `--algorithm grpo` for token-level
importance sampling and `--algorithm gspo` for sequence-level importance
sampling. Treat it as a compact reproducibility example; for larger models or
newer Unsloth releases, port the same reward tests and reporting contract into
the current recommended runtime.

Example command inside a compatible Unsloth GPU environment:

```sh
python unsloth_grpo_gspo_reasoning_experiment.py \
  --run-experiment \
  --artifact-dir runs/unsloth_grpo_gspo_reasoning \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --max-steps 5 \
  --num-generations 4
```

The run writes `unsloth_grpo_gspo_summary.json` and
`unsloth_grpo_gspo_outputs.jsonl` under the selected artifact directory.
