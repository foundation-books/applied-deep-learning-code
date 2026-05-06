#!/usr/bin/env python3
"""Prompt-only LLM benchmark scaffold for the LLM chapter homework.

The script is import-safe on machines without PyTorch or Transformers. Repository
checks can compile it and inspect optional dependencies. A real generation run
requires a compatible model checkpoint and runtime environment.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Iterable


RUNTIME_PACKAGES = ["torch", "transformers", "accelerate"]
SYSTEM_PROMPT = "You are a concise applied deep learning course assistant."
RUBRIC_COMPONENTS = [
    ("technical_correctness", 4.0),
    ("course_terminology", 2.0),
    ("applied_advice", 2.0),
    ("concision", 1.0),
    ("safety", 1.0),
]


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_status() -> dict[str, bool]:
    return {name: package_available(name) for name in RUNTIME_PACKAGES}


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def built_in_prompts() -> list[dict[str, str]]:
    return [
        {
            "question": "How do I improve validation accuracy in an image classifier?",
            "reference_answer": (
                "Check the validation split and baseline first. Then change one "
                "factor at a time, such as augmentation, learning rate, model size, "
                "or input resolution. Keep the final test set untouched."
            ),
        },
        {
            "question": "Why should I not tune hyperparameters on the test set?",
            "reference_answer": (
                "The test set is meant to estimate performance after model selection. "
                "Repeated tuning on it turns it into another validation set."
            ),
        },
        {
            "question": "What does the KV cache speed up during generation?",
            "reference_answer": (
                "It stores keys and values for previous tokens so the model does not "
                "recompute the whole prefix at every autoregressive step."
            ),
        },
    ]


def format_prompt(question: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    return (
        f"{system_prompt}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def chat_messages(
    question: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def has_chat_template(tokenizer: object) -> bool:
    return bool(getattr(tokenizer, "chat_template", None)) and callable(
        getattr(tokenizer, "apply_chat_template", None)
    )


def prompt_text(
    record: dict[str, str],
    tokenizer: object | None = None,
    use_chat_template: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[str, str]:
    if record.get("prompt"):
        return record["prompt"], "raw_prompt"
    if use_chat_template and tokenizer is not None and has_chat_template(tokenizer):
        text = tokenizer.apply_chat_template(
            chat_messages(record["question"], system_prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
        return str(text), "chat_template"
    return format_prompt(record["question"], system_prompt), "plain_prompt"


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if "prompt" in record:
                rows.append(
                    {
                        "prompt": str(record["prompt"]),
                        "question": str(record.get("question", "")),
                        "reference_answer": str(
                            record.get("reference_answer", record.get("answer", ""))
                        ),
                    }
                )
                continue
            if "question" not in record:
                raise ValueError(f"{path}:{line_number} must contain prompt or question")
            rows.append(
                {
                    "question": str(record["question"]),
                    "reference_answer": str(record.get("reference_answer", record.get("answer", ""))),
                }
            )
    return rows


def load_prompts(path: Path | None) -> list[dict[str, str]]:
    return read_jsonl(path) if path else built_in_prompts()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_transformer_runtime():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


def cuda_summary(torch_module: object) -> dict[str, object]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "device_name": cuda.get_device_name(0),
        "max_memory_allocated_bytes": int(cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(cuda.max_memory_reserved()),
    }


def count_tokens(tokenizer: object, text: str, prompt_format: str) -> int:
    encoded = tokenizer(text, add_special_tokens=prompt_format != "chat_template")
    return int(len(encoded["input_ids"]))


def preview(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts)
    for index, record in enumerate(prompts, start=1):
        text, prompt_format = prompt_text(record, system_prompt=args.system_prompt)
        print(f"[{index}] ({prompt_format}) {text}")
        if record.get("reference_answer"):
            print(f"reference: {record['reference_answer']}")


def estimate_tokens(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts)
    _, _, AutoTokenizer = load_transformer_runtime()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    rows = []
    for index, record in enumerate(prompts, start=1):
        text, prompt_format = prompt_text(
            record,
            tokenizer,
            args.use_chat_template,
            args.system_prompt,
        )
        rows.append(
            {
                "index": index,
                "question": record.get("question", ""),
                "prompt_format": prompt_format,
                "prompt_token_count": count_tokens(tokenizer, text, prompt_format),
            }
        )
    write_jsonl(args.output, rows)
    print(f"Wrote token counts to {args.output}")


def parse_score(value: str, field: str, row_number: int) -> float:
    try:
        score = float(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} must be numeric, got {value!r}") from exc
    return score


def aggregate_scores(args: argparse.Namespace) -> None:
    if args.scores_csv is None:
        raise SystemExit("--aggregate-scores requires --scores-csv")

    rows: list[dict[str, object]] = []
    with args.scores_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field, _ in RUBRIC_COMPONENTS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{args.scores_csv} is missing score columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            component_scores: dict[str, float] = {}
            for field, max_score in RUBRIC_COMPONENTS:
                score = parse_score(str(row.get(field, "")).strip(), field, row_number)
                if score < 0 or score > max_score:
                    raise ValueError(
                        f"row {row_number}: {field}={score} is outside the expected 0..{max_score} range"
                    )
                component_scores[field] = score
            total = sum(component_scores.values())
            rows.append(
                {
                    "index": row.get("index") or row.get("prompt_index") or len(rows) + 1,
                    "question": row.get("question", ""),
                    "notes": row.get("notes", row.get("error_notes", "")),
                    "scores": component_scores,
                    "total": total,
                }
            )

    if not rows:
        raise ValueError(f"{args.scores_csv} contains no scored rows")

    component_averages = {
        field: sum(float(row["scores"][field]) for row in rows) / len(rows)
        for field, _ in RUBRIC_COMPONENTS
    }
    total_scores = [float(row["total"]) for row in rows]
    payload = {
        "scores_csv": str(args.scores_csv),
        "generation_output": str(args.output),
        "metadata_output": str(args.metadata_output),
        "rubric_max_total": sum(max_score for _, max_score in RUBRIC_COMPONENTS),
        "records": len(rows),
        "average_total": sum(total_scores) / len(total_scores),
        "min_total": min(total_scores),
        "max_total": max(total_scores),
        "component_averages": component_averages,
        "rows": rows,
    }
    write_json(args.score_output, payload)
    print(f"Wrote score summary to {args.score_output}")


def generate(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts)
    torch, AutoModelForCausalLM, AutoTokenizer = load_transformer_runtime()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device_map,
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
    )
    if (
        getattr(tokenizer, "pad_token_id", None) is None
        and getattr(tokenizer, "eos_token_id", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    outputs: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, record in enumerate(prompts, start=1):
        text, prompt_format = prompt_text(
            record,
            tokenizer,
            args.use_chat_template,
            args.system_prompt,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=prompt_format != "chat_template",
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        first_started = time.perf_counter()
        with torch.inference_mode():
            first = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        first_token_seconds = time.perf_counter() - first_started

        generation_started = time.perf_counter()
        generation_kwargs = {
            **inputs,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p
        with torch.inference_mode():
            generated = model.generate(**generation_kwargs)
        generation_seconds = time.perf_counter() - generation_started
        new_tokens = int(generated.shape[-1] - prompt_tokens)
        generated_text = tokenizer.decode(generated[0, prompt_tokens:], skip_special_tokens=True)
        outputs.append(
            {
                "index": index,
                "question": record.get("question", ""),
                "reference_answer": record.get("reference_answer", ""),
                "prompt": text,
                "prompt_format": prompt_format,
                "prompt_token_count": prompt_tokens,
                "generated_token_count": new_tokens,
                "first_token_seconds": first_token_seconds,
                "full_generation_seconds": generation_seconds,
                "output_tokens_per_second": (
                    new_tokens / generation_seconds if generation_seconds else None
                ),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "generated_text": generated_text.strip(),
            }
        )

    write_jsonl(args.output, outputs)
    metadata = {
        "model_name": args.model_name,
        "prompt_count": len(prompts),
        "system_prompt": args.system_prompt,
        "use_chat_template": args.use_chat_template,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "wall_clock_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": dependency_versions(),
        "cuda": cuda_summary(torch),
    }
    write_json(args.metadata_output, metadata)
    print(f"Wrote generations to {args.output}")
    print(f"Wrote metadata to {args.metadata_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-deps", action="store_true")
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--estimate-tokens", action="store_true")
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--aggregate-scores", action="store_true")
    parser.add_argument("--allow-missing-deps", action="store_true")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/llm_prompt_outputs.jsonl"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("runs/llm_prompt_metadata.json"),
    )
    parser.add_argument("--scores-csv", type=Path)
    parser.add_argument(
        "--score-output",
        type=Path,
        default=Path("runs/llm_prompt_score_summary.json"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_deps:
        status = dependency_status()
        print(
            json.dumps(
                {"available": status, "versions": dependency_versions()},
                indent=2,
                sort_keys=True,
            )
        )
        missing = [name for name, available in status.items() if not available]
        if missing and not args.allow_missing_deps:
            raise SystemExit(f"Missing optional runtime packages: {', '.join(missing)}")
        return
    if args.preview:
        preview(args)
        return
    if args.estimate_tokens:
        estimate_tokens(args)
        return
    if args.generate:
        generate(args)
        return
    if args.aggregate_scores:
        aggregate_scores(args)
        return
    print("Use --preview, --estimate-tokens, --generate, --aggregate-scores, or --check-deps.")


if __name__ == "__main__":
    main()
