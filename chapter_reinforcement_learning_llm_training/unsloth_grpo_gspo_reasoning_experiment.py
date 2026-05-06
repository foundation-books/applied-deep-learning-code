#!/usr/bin/env python3
"""Small Unsloth GRPO/GSPO experiment for the RL chapter.

This runner is intentionally narrow: it uses a small arithmetic prompt split
and trains through Unsloth and TRL's ``GRPOTrainer``. GSPO is selected by
setting TRL's
``importance_sampling_level`` to ``"sequence"``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


RUNTIME_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "peft",
    "trl",
    "unsloth",
)

STRICT_ANSWER_PATTERN = re.compile(r"^\s*<answer>\s*([-+]?\d+)\s*</answer>\s*$", re.IGNORECASE)

SYSTEM_PROMPT = (
    "Return exactly one XML tag in this format: <answer>number</answer>. "
    "Do not include prose or reasoning."
)

TRAIN_RECORDS = [
    {"question": "Compute 14 + 9.", "final_answer": "23"},
    {"question": "Compute 18 + 7.", "final_answer": "25"},
    {"question": "Compute 21 + 16.", "final_answer": "37"},
    {"question": "Compute 27 + 15.", "final_answer": "42"},
    {"question": "Compute 34 + 8.", "final_answer": "42"},
    {"question": "Compute 43 + 19.", "final_answer": "62"},
    {"question": "Compute 58 - 23.", "final_answer": "35"},
    {"question": "Compute 72 - 29.", "final_answer": "43"},
    {"question": "Compute 91 - 47.", "final_answer": "44"},
    {"question": "Compute 6 * 7.", "final_answer": "42"},
    {"question": "Compute 8 * 9.", "final_answer": "72"},
    {"question": "Compute 11 * 6.", "final_answer": "66"},
    {"question": "Compute 12 * 4.", "final_answer": "48"},
    {"question": "Compute 13 + 28.", "final_answer": "41"},
    {"question": "Compute 75 - 38.", "final_answer": "37"},
    {"question": "Compute 9 * 5.", "final_answer": "45"},
]

VALIDATION_RECORDS = [
    {"question": "Compute 17 + 25.", "final_answer": "42"},
    {"question": "Compute 29 + 34.", "final_answer": "63"},
    {"question": "Compute 84 - 39.", "final_answer": "45"},
    {"question": "Compute 96 - 58.", "final_answer": "38"},
    {"question": "Compute 7 * 8.", "final_answer": "56"},
    {"question": "Compute 12 * 7.", "final_answer": "84"},
    {"question": "Compute 46 + 27.", "final_answer": "73"},
    {"question": "Compute 15 * 9.", "final_answer": "135"},
]


@dataclass(frozen=True)
class RewardBreakdown:
    correctness: float
    format_score: float
    length_penalty: float

    @property
    def total(self) -> float:
        return self.correctness + self.format_score + self.length_penalty


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def dependency_status() -> dict[str, bool]:
    return {name: package_available(name) for name in RUNTIME_PACKAGES}


def patch_vllm_guided_decoding_import() -> None:
    """Allow TRL GRPO imports in vLLM images where this class moved."""
    try:
        import vllm.sampling_params as sampling_params
    except Exception:
        return
    if hasattr(sampling_params, "GuidedDecodingParams"):
        return

    class GuidedDecodingParams:  # pragma: no cover - only used for import compatibility.
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    sampling_params.GuidedDecodingParams = GuidedDecodingParams


def require_deps() -> dict[str, Any]:
    import unsloth
    from unsloth import FastLanguageModel, is_bfloat16_supported

    patch_vllm_guided_decoding_import()

    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    return {
        "Dataset": Dataset,
        "FastLanguageModel": FastLanguageModel,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
        "is_bfloat16_supported": is_bfloat16_supported,
        "torch": torch,
        "unsloth": unsloth,
    }


def cuda_environment(torch_module: object | None = None) -> dict[str, object]:
    summary: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
    }
    if torch_module is None:
        return summary

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        summary["cuda"] = {"available": False}
        return summary
    cuda_summary: dict[str, object] = {"available": bool(cuda.is_available())}
    if cuda.is_available():
        cuda_summary["device_count"] = int(cuda.device_count())
        devices: list[dict[str, object]] = []
        for index in range(cuda.device_count()):
            props = cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": cuda.get_device_name(index),
                    "capability": list(cuda.get_device_capability(index)),
                    "total_memory_bytes": int(props.total_memory),
                }
            )
        cuda_summary["devices"] = devices
    summary["cuda"] = cuda_summary
    return summary

def set_seed(seed: int, torch_module: object | None = None) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch_module is not None:
        torch_module.manual_seed(seed)
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.manual_seed_all(seed)


def extract_answer(completion: str) -> str | None:
    match = STRICT_ANSWER_PATTERN.fullmatch(completion)
    if match is None:
        return None
    return match.group(1)


def has_strict_answer_format(completion: str) -> bool:
    return extract_answer(completion) is not None


def score_completion(
    completion: str,
    expected_answer: str,
    max_tokens: int,
    generated_tokens: int | None = None,
) -> RewardBreakdown:
    extracted = extract_answer(completion)
    token_count = generated_tokens if generated_tokens is not None else len(completion.split())
    extra_tokens = max(0, token_count - max_tokens)
    return RewardBreakdown(
        correctness=1.0 if extracted == expected_answer else 0.0,
        format_score=0.2 if has_strict_answer_format(completion) else 0.0,
        length_penalty=-0.01 * extra_tokens,
    )


def completion_texts(completions: list[object]) -> list[str]:
    texts: list[str] = []
    for completion in completions:
        if isinstance(completion, list) and completion:
            first = completion[0]
            if isinstance(first, dict):
                texts.append(str(first.get("content", "")))
            else:
                texts.append(str(first))
        else:
            texts.append(str(completion))
    return texts


def correctness_reward_func(completions: list[object], answer: list[str], **_: object) -> list[float]:
    return [1.0 if extract_answer(text) == expected else 0.0 for text, expected in zip(completion_texts(completions), answer)]


def strict_format_reward_func(completions: list[object], **_: object) -> list[float]:
    return [0.2 if has_strict_answer_format(text) else 0.0 for text in completion_texts(completions)]


def length_reward_func(completions: list[object], completion_ids: list[object] | None = None, **_: object) -> list[float]:
    texts = completion_texts(completions)
    if completion_ids is None:
        return [-0.01 * max(0, len(text.split()) - 10) for text in texts]
    penalties: list[float] = []
    for ids, text in zip(completion_ids, texts):
        try:
            token_count = len(ids)
        except TypeError:
            token_count = len(text.split())
        penalties.append(-0.01 * max(0, token_count - 10))
    return penalties


def to_dataset(records: list[dict[str, str]], dataset_class: object) -> object:
    return dataset_class.from_list(
        [
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": record["question"]},
                ],
                "answer": record["final_answer"],
            }
            for record in records
        ]
    )


def cuda_sync(torch_module: object) -> None:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.synchronize()


def reset_cuda_peak(torch_module: object) -> None:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.reset_peak_memory_stats()


def peak_cuda_memory(torch_module: object) -> dict[str, object]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "max_memory_allocated_bytes": int(cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(cuda.max_memory_reserved()),
    }


def input_device(model: object) -> object:
    return next(model.parameters()).device


def generate_one(
    model: object,
    tokenizer: object,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    torch_module: object,
) -> tuple[str, int, float]:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
    )
    device = input_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_width = int(encoded["input_ids"].shape[1])

    cuda_sync(torch_module)
    started = time.perf_counter()
    with torch_module.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_completion_length,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    cuda_sync(torch_module)

    generated_ids = output_ids[0][input_width:]
    pad_id = getattr(tokenizer, "pad_token_id", None)
    generated_count = int(generated_ids.numel()) if pad_id is None else int((generated_ids != pad_id).sum().item())
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text, generated_count, time.perf_counter() - started


def summarize_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    rewards = [float(row["reward"]["total"]) for row in rows]
    exact = [float(row["reward"]["correctness"]) for row in rows]
    valid = [1.0 if bool(row["strict_answer_format"]) else 0.0 for row in rows]
    generated_tokens = [float(row["generated_token_count"]) for row in rows]
    generation_seconds = [float(row["generation_seconds"]) for row in rows]
    total_seconds = sum(generation_seconds)
    return {
        "records": len(rows),
        "average_total_reward": sum(rewards) / len(rewards),
        "exact_answer_accuracy": sum(exact) / len(exact),
        "format_valid_rate": sum(valid) / len(valid),
        "invalid_output_rate": 1.0 - (sum(valid) / len(valid)),
        "average_generated_tokens": sum(generated_tokens) / len(generated_tokens),
        "tokens_per_second": sum(generated_tokens) / total_seconds if total_seconds > 0 else None,
    }


def evaluate(
    model: object,
    tokenizer: object,
    records: list[dict[str, str]],
    args: argparse.Namespace,
    torch_module: object,
    phase: str,
    fast_language_model: object,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    fast_language_model.for_inference(model)
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["question"]},
        ]
        completion, generated_count, seconds = generate_one(model, tokenizer, messages, args, torch_module)
        reward = score_completion(
            completion,
            record["final_answer"],
            args.reward_max_tokens,
            generated_tokens=generated_count,
        )
        rows.append(
            {
                "phase": phase,
                "index": index,
                "question": record["question"],
                "expected": record["final_answer"],
                "generated_text": completion,
                "extracted_answer": extract_answer(completion),
                "strict_answer_format": has_strict_answer_format(completion),
                "generated_token_count": generated_count,
                "generation_seconds": seconds,
                "reward": {"total": reward.total, **asdict(reward)},
            }
        )
    return rows, summarize_metrics(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def model_revision_from_path(model_name: str) -> str | None:
    parts = Path(model_name).parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def load_model(args: argparse.Namespace, deps: dict[str, Any]) -> tuple[object, object]:
    fast_language_model = deps["FastLanguageModel"]
    model_kwargs = {
        "model_name": args.model_name,
        "max_seq_length": args.max_seq_length,
        "dtype": None,
        "load_in_4bit": args.load_in_4bit,
        "load_in_16bit": not args.load_in_4bit,
        "fast_inference": False,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.model_revision:
        model_kwargs["revision"] = args.model_revision
    model, tokenizer = fast_language_model.from_pretrained(**model_kwargs)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    modules = [module.strip() for module in args.target_modules.split(",") if module.strip()]
    model = fast_language_model.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    return model, tokenizer


def trainable_parameter_summary(model: object) -> dict[str, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if bool(getattr(parameter, "requires_grad", False)):
            trainable += count
    return {"total_parameters": total, "trainable_parameters": trainable}


def run_experiment(args: argparse.Namespace) -> None:
    deps = require_deps()
    torch_module = deps["torch"]
    set_seed(args.seed, torch_module)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir = args.artifact_dir / "trainer_output"

    started = time.perf_counter()
    model, tokenizer = load_model(args, deps)

    reset_cuda_peak(torch_module)
    baseline_rows, baseline_metrics = evaluate(
        model,
        tokenizer,
        VALIDATION_RECORDS,
        args,
        torch_module,
        "prompt_only_baseline",
        deps["FastLanguageModel"],
    )
    baseline_peak = peak_cuda_memory(torch_module)

    deps["FastLanguageModel"].for_training(model)
    bf16 = bool(deps["is_bfloat16_supported"]())
    importance_sampling_level = "sequence" if args.algorithm == "gspo" else "token"
    training_args = deps["GRPOConfig"](
        output_dir=str(output_dir),
        use_vllm=False,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim=args.optim,
        logging_steps=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.max_steps,
        max_grad_norm=args.max_grad_norm,
        report_to="none",
        bf16=bf16,
        fp16=not bf16,
        temperature=args.temperature,
        top_p=args.top_p,
        importance_sampling_level=importance_sampling_level,
        loss_type=args.loss_type,
        seed=args.seed,
    )
    trainer = deps["GRPOTrainer"](
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            strict_format_reward_func,
            length_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=to_dataset(TRAIN_RECORDS, deps["Dataset"]),
    )

    reset_cuda_peak(torch_module)
    train_started = time.perf_counter()
    train_result = trainer.train()
    cuda_sync(torch_module)
    training_seconds = time.perf_counter() - train_started
    training_peak = peak_cuda_memory(torch_module)

    reset_cuda_peak(torch_module)
    trained_rows, trained_metrics = evaluate(
        model,
        tokenizer,
        VALIDATION_RECORDS,
        args,
        torch_module,
        f"{args.algorithm}_lora",
        deps["FastLanguageModel"],
    )
    trained_peak = peak_cuda_memory(torch_module)

    if args.save_adapter:
        adapter_dir = args.artifact_dir / f"{args.algorithm}_lora_adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
    else:
        adapter_dir = None

    output_path = args.artifact_dir / "unsloth_grpo_gspo_outputs.jsonl"
    summary_path = args.artifact_dir / "unsloth_grpo_gspo_summary.json"
    write_jsonl(output_path, [*baseline_rows, *trained_rows])
    write_json(
        summary_path,
        {
            "run_kind": "unsloth_grpo_gspo_reasoning",
            "algorithm": args.algorithm,
            "importance_sampling_level": importance_sampling_level,
            "model_name": args.report_model_name,
            "model_load_path": args.model_name,
            "model_revision": args.model_revision or model_revision_from_path(args.model_name),
            "seed": args.seed,
            "train_records": len(TRAIN_RECORDS),
            "validation_records": len(VALIDATION_RECORDS),
            "system_prompt": SYSTEM_PROMPT,
            "max_seq_length": args.max_seq_length,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_generations": args.num_generations,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "loss_type": args.loss_type,
            "max_grad_norm": args.max_grad_norm,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules,
            "quantization": "4-bit" if args.load_in_4bit else "none",
            "parameters": trainable_parameter_summary(model),
            "baseline_metrics": baseline_metrics,
            "trained_metrics": trained_metrics,
            "trainer_metrics": getattr(train_result, "metrics", {}),
            "training_summary": {
                "training_seconds": training_seconds,
                "steps": args.max_steps,
                "mean_step_seconds": training_seconds / args.max_steps if args.max_steps else None,
            },
            "peak_memory": {
                "baseline": baseline_peak,
                "training": training_peak,
                "trained_evaluation": trained_peak,
            },
            "environment": cuda_environment(torch_module),
            "wall_clock_seconds": time.perf_counter() - started,
            "output_file": str(output_path),
            "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
        },
    )
    print(f"Wrote outputs to {output_path}")
    print(f"Wrote summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-deps", action="store_true")
    mode.add_argument("--run-experiment", action="store_true")
    parser.add_argument("--allow-missing-deps", action="store_true")
    parser.add_argument("--algorithm", choices=["grpo", "gspo"], default="grpo")
    parser.add_argument("--artifact-dir", type=Path, default=Path("runs/unsloth_grpo_gspo_reasoning"))
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--report-model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--max-prompt-length", type=int, default=192)
    parser.add_argument("--max-completion-length", type=int, default=10)
    parser.add_argument("--reward-max-tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--loss-type", default="grpo")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--save-adapter", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_deps:
        status = dependency_status()
        print(json.dumps({"available": status, "versions": package_versions()}, indent=2, sort_keys=True))
        if not args.allow_missing_deps and not all(status.values()):
            missing = [name for name, available in status.items() if not available]
            raise SystemExit("Missing dependencies: " + ", ".join(missing))
        return
    if args.run_experiment:
        run_experiment(args)
        return
    print(
        json.dumps(
            {
                "train_records": TRAIN_RECORDS,
                "validation_records": VALIDATION_RECORDS,
                "system_prompt": SYSTEM_PROMPT,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("Use --check-deps for smoke checks or --run-experiment inside a compatible GPU environment.")


if __name__ == "__main__":
    main()
