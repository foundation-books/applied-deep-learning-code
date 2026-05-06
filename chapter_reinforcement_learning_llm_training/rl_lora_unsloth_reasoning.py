"""Reward and data scaffold for the RL-for-LLM-training homework.

The file is intentionally conservative. It can be run on a CPU-only machine to
test reward functions and JSONL data format. A real LoRA/QLoRA reinforcement
learning run should start from current Unsloth GRPO or GSPO notebooks.
"""

from __future__ import annotations

import argparse
import importlib.util
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Iterable

from reward_functions import score_completion


OPTIONAL_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "trl",
    "unsloth",
)

SAMPLE_DATA_PATH = Path(__file__).with_name("sample_reasoning_prompts.jsonl")


def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def metadata_template() -> dict[str, object]:
    return {
        "run": {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_revision": None,
            "seed": 7,
            "hardware": platform.platform(),
            "package_versions": {
                "python": sys.version.split()[0],
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "trl": package_version("trl"),
                "unsloth": package_version("unsloth"),
            },
        },
        "data": {
            "train_path": "train.jsonl",
            "validation_path": "validation.jsonl",
            "held_out_prompt_count": None,
            "prompt_template": "Answer with exactly one <answer>...</answer> tag.",
        },
        "lora": {
            "rank": 8,
            "alpha": 8,
            "target_modules": None,
            "dropout": 0.0,
            "quantization": "4-bit if GPU memory requires it; otherwise none",
        },
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_prompt_length": None,
            "max_completion_length": None,
            "num_generations": None,
        },
        "optimization": {
            "algorithm": "GRPO or GSPO",
            "learning_rate": None,
            "batch_size": None,
            "gradient_accumulation_steps": None,
            "training_steps": None,
            "kl_setting": None,
        },
        "reward": {
            "reward_code_path": str(Path(__file__).with_name("reward_functions.py")),
            "correctness_weight": 1.0,
            "format_weight": 0.2,
            "length_penalty_per_extra_token": -0.01,
            "unit_test_command": "python rl_lora_unsloth_reasoning.py --unit-test-rewards",
        },
        "measurements": {
            "prompt_only_validation_score": None,
            "rl_validation_score": None,
            "invalid_output_rate": None,
            "average_completion_length": None,
            "tokens_per_second": None,
            "peak_gpu_memory": None,
            "training_time": None,
        },
        "artifacts": {
            "adapter_path": None,
            "baseline_outputs_path": None,
            "trained_outputs_path": None,
            "ablation_summary": None,
        },
    }


def iter_jsonl(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("question", "final_answer"):
                if key not in record:
                    raise ValueError(f"{path}:{line_no} missing required key {key!r}")
            yield record


def write_jsonl(path: Path, records: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_sample_records() -> list[dict[str, str]]:
    return list(iter_jsonl(SAMPLE_DATA_PATH))


def check_optional_dependencies(allow_missing: bool) -> int:
    missing = []
    for package in OPTIONAL_PACKAGES:
        available = importlib.util.find_spec(package) is not None
        print(f"{package}: {'available' if available else 'missing'}")
        if not available:
            missing.append(package)

    if missing and not allow_missing:
        print("Missing optional packages for a real Unsloth RL run:", ", ".join(missing))
        return 1
    return 0


def unit_test_rewards() -> None:
    correct = score_completion("<answer>42</answer>", "42")
    wrong = score_completion("<answer>41</answer>", "42")
    missing = score_completion("The answer is 42.", "42")
    long = score_completion("<answer>42</answer> " + "extra " * 90, "42", max_tokens=10)
    multiple_tags = score_completion("<answer>42</answer><answer>41</answer>", "42")
    prose_around_tag = score_completion("The answer is <answer>42</answer>.", "42")
    contradictory = score_completion("The answer is 41. <answer>42</answer>", "42")
    empty = score_completion("<answer></answer>", "42")
    expression = score_completion("<answer>40+2</answer>", "42")

    assert correct.correctness == 1.0
    assert correct.format_score == 0.2
    assert wrong.correctness == 0.0
    assert wrong.format_score == 0.2
    assert missing.correctness == 0.0
    assert missing.format_score == 0.0
    assert long.length_penalty < 0.0
    for exploit in (multiple_tags, prose_around_tag, contradictory, empty):
        assert exploit.correctness == 0.0
        assert exploit.format_score == 0.0
    assert expression.correctness == 0.0
    assert expression.format_score == 0.2
    print("reward unit tests passed")


def write_sample_data(output_dir: Path) -> None:
    records = load_sample_records()
    train = records[2:]
    validation = records[:2]
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "validation.jsonl", validation)
    print(f"wrote {len(train)} training records to {output_dir / 'train.jsonl'}")
    print(f"wrote {len(validation)} validation records to {output_dir / 'validation.jsonl'}")


def write_metadata_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_template(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote metadata template to {path}")


def score_jsonl(path: Path) -> None:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"{path} contains no records")

    total = 0.0
    for record in rows:
        completion = record.get("answer", "")
        score = score_completion(completion, record["final_answer"])
        total += score.total
        print(
            json.dumps(
                {
                    "question": record["question"],
                    "expected": record["final_answer"],
                    "score": score.to_dict(),
                },
                sort_keys=True,
            )
        )
    print(f"average_total_reward: {total / len(rows):.4f}")


def print_training_note(use_gspo: bool) -> None:
    mode = "GSPO" if use_gspo else "GRPO"
    print(
        f"Use a versioned Unsloth {mode} notebook for the GPU run. "
        "Reuse this reward function, record the exact model and package versions, "
        "write the metadata template, save the LoRA adapter, and evaluate held-out prompts with fixed decoding."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-deps", action="store_true", help="Check optional Unsloth training dependencies.")
    parser.add_argument("--allow-missing-deps", action="store_true", help="Return success even when optional packages are missing.")
    parser.add_argument("--unit-test-rewards", action="store_true", help="Run reward-function unit tests.")
    parser.add_argument("--write-sample-data", type=Path, help="Write train.jsonl and validation.jsonl under this directory.")
    parser.add_argument("--write-metadata-template", type=Path, help="Write a JSON metadata template for a GRPO/GSPO run.")
    parser.add_argument("--score-jsonl", type=Path, help="Score records that contain answer and final_answer fields.")
    parser.add_argument("--training-note", action="store_true", help="Print the intended Unsloth training workflow.")
    parser.add_argument("--use-gspo", action="store_true", help="Mention GSPO rather than GRPO in the training note.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exit_code = 0

    if args.check_deps:
        exit_code = max(exit_code, check_optional_dependencies(args.allow_missing_deps))
    if args.unit_test_rewards:
        unit_test_rewards()
    if args.write_sample_data:
        write_sample_data(args.write_sample_data)
    if args.write_metadata_template:
        write_metadata_template(args.write_metadata_template)
    if args.score_jsonl:
        score_jsonl(args.score_jsonl)
    if args.training_note:
        print_training_note(args.use_gspo)

    if not any(
        (
            args.check_deps,
            args.unit_test_rewards,
            args.write_sample_data,
            args.write_metadata_template,
            args.score_jsonl,
            args.training_note,
        )
    ):
        print("Use --help to see reward tests, sample-data writing, and dependency checks.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
