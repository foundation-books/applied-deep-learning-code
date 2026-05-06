"""LoRA fine-tuning scaffold for the LoRA/QLoRA adaptation homework.

The script is designed to be importable and syntax-checkable on machines that
do not have a GPU or Unsloth installed. Use --check-deps to inspect optional
dependencies. Use --train only inside a compatible Unsloth environment.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import time
from pathlib import Path
from typing import Iterable


REQUIRED_FOR_TRAINING = [
    "torch",
    "datasets",
    "peft",
    "transformers",
    "trl",
    "unsloth",
]


def dependency_status() -> dict[str, bool]:
    """Return whether each optional training dependency can be imported."""

    return {
        name: importlib.util.find_spec(name) is not None
        for name in REQUIRED_FOR_TRAINING
    }


def dependency_versions() -> dict[str, str | None]:
    """Return installed package versions when package metadata is available."""

    versions: dict[str, str | None] = {}
    for name in REQUIRED_FOR_TRAINING:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def built_in_records() -> list[dict[str, str]]:
    """Small demonstration records; not enough for a real fine-tune."""

    return [
        {
            "question": "How do I improve validation accuracy in an image classifier?",
            "answer": (
                "Start by checking the validation split and baseline. Then change "
                "one factor at a time, such as augmentation, learning rate, model "
                "size, or input resolution. Keep the final test set untouched."
            ),
        },
        {
            "question": "Why should I not tune hyperparameters on the test set?",
            "answer": (
                "The test set is meant to estimate performance after model selection. "
                "If you use it repeatedly during tuning, it becomes another validation "
                "set and the final number is no longer an honest estimate."
            ),
        },
        {
            "question": "When is LoRA a better first step than full fine-tuning?",
            "answer": (
                "LoRA is usually better when the base model is already useful and you "
                "need a cheaper adaptation of style, format, or domain behavior. It "
                "trains far fewer parameters and uses less optimizer memory."
            ),
        },
    ]


def read_jsonl(path: Path) -> list[dict[str, str]]:
    """Load records with question and answer fields from a JSONL file."""

    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if "question" not in record or "answer" not in record:
                raise ValueError(
                    f"{path}:{line_number} must contain question and answer fields"
                )
            records.append(
                {
                    "question": str(record["question"]),
                    "answer": str(record["answer"]),
                }
            )
    return records


def format_prompt(question: str, answer: str | None = None) -> str:
    """Format one course-assistant example as plain instruction text."""

    prefix = (
        "You are a concise applied deep learning course assistant.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    if answer is None:
        return prefix
    return f"{prefix} {answer}"


def format_with_optional_chat_template(
    tokenizer: object,
    question: str,
    answer: str | None,
    use_chat_template: bool,
) -> str:
    """Format an example with the tokenizer chat template when available."""

    chat_template = getattr(tokenizer, "chat_template", None)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if use_chat_template and chat_template and callable(apply_chat_template):
        messages = [
            {
                "role": "system",
                "content": "You are a concise applied deep learning course assistant.",
            },
            {"role": "user", "content": question},
        ]
        if answer is not None:
            messages.append({"role": "assistant", "content": answer})
        return str(
            apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=answer is None,
            )
        )
    return format_prompt(question, answer)


def records_to_text(
    records: Iterable[dict[str, str]],
    tokenizer: object,
    use_chat_template: bool,
) -> list[dict[str, str]]:
    """Convert question-answer records to the text field expected by SFTTrainer."""

    return [
        {
            "text": format_with_optional_chat_template(
                tokenizer,
                record["question"],
                record["answer"],
                use_chat_template,
            )
        }
        for record in records
    ]


def write_json(path: Path, payload: object) -> None:
    """Write a JSON artifact with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write JSONL rows for generated validation outputs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def model_provenance(args: argparse.Namespace) -> dict[str, str]:
    """Return base-model provenance fields for adapter release review."""

    model_card_url = args.model_card_url
    if not model_card_url and "/" in args.model_name and not Path(args.model_name).exists():
        model_card_url = f"https://huggingface.co/{args.model_name}"
    return {
        "model_name": args.model_name,
        "model_revision": args.model_revision or "not-recorded",
        "model_license": args.model_license or "not-recorded",
        "model_card_url": model_card_url or "not-recorded",
        "adapter_release_note": (
            "Before sharing an adapter, review the base model card, license, "
            "training-data terms, and the dataset used for this adaptation."
        ),
    }


def trainable_parameter_count(model: object) -> int | None:
    """Count trainable parameters when the model exposes PyTorch parameters."""

    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    return int(sum(param.numel() for param in parameters() if getattr(param, "requires_grad", False)))


def cuda_memory_summary(torch_module: object) -> dict[str, object]:
    """Return a small CUDA memory summary without requiring CUDA."""

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "device_name": cuda.get_device_name(0),
        "max_memory_allocated_bytes": int(cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(cuda.max_memory_reserved()),
    }


def load_records(path: Path | None) -> list[dict[str, str]]:
    """Load JSONL records or fall back to the tiny built-in demonstration set."""

    return read_jsonl(path) if path else built_in_records()


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[object, object]:
    """Load an Unsloth model and tokenizer for either inference or training."""

    from unsloth import FastLanguageModel

    kwargs: dict[str, object] = {
        "model_name": args.model_name,
        "max_seq_length": args.max_seq_length,
        "dtype": None,
        "load_in_4bit": args.load_in_4bit,
    }
    if args.model_revision:
        kwargs["revision"] = args.model_revision
    return FastLanguageModel.from_pretrained(**kwargs)


def generate_one(
    model: object,
    tokenizer: object,
    prompt: str,
    args: argparse.Namespace,
    torch_module: object,
) -> str:
    """Generate one answer for a validation prompt."""

    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs: dict[str, object] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch_module.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_length:]
    return str(tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()


def run_prompt_only_baseline(args: argparse.Namespace) -> None:
    """Generate held-out validation answers without updating model weights."""

    missing = [name for name, present in dependency_status().items() if not present]
    if missing:
        raise SystemExit(
            "Missing inference dependencies: "
            + ", ".join(missing)
            + ". Install Unsloth and related packages in a compatible environment."
        )

    import torch
    from unsloth import FastLanguageModel

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    records = load_records(args.validation_data)
    started = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(args)
    FastLanguageModel.for_inference(model)

    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        prompt = format_with_optional_chat_template(
            tokenizer,
            record["question"],
            answer=None,
            use_chat_template=args.use_chat_template,
        )
        generated = generate_one(model, tokenizer, prompt, args, torch)
        rows.append(
            {
                "index": index,
                "question": record["question"],
                "reference_answer": record.get("answer", ""),
                "generated_answer": generated,
            }
        )

    elapsed = time.perf_counter() - started
    write_jsonl(args.baseline_output, rows)
    write_json(
        args.output_dir / f"{args.run_name}_baseline_metadata.json",
        {
            "run_kind": "baseline",
            "run_name": args.run_name,
            "model_name": args.model_name,
            "model_provenance": model_provenance(args),
            "records": len(rows),
            "max_seq_length": args.max_seq_length,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "use_chat_template": args.use_chat_template,
            "wall_clock_seconds": elapsed,
            "dependencies": dependency_versions(),
            "gpu_memory": cuda_memory_summary(torch),
            "output_file": str(args.baseline_output),
        },
    )
    print(f"Wrote baseline generations to {args.baseline_output}")


def inspect_tokenized_batch(args: argparse.Namespace) -> None:
    """Write a small tokenization inspection artifact for training records."""

    if importlib.util.find_spec("transformers") is None:
        raise SystemExit("Missing transformers dependency; install it before tokenization inspection.")

    from transformers import AutoTokenizer

    records = load_records(args.data)[: args.inspect_examples]
    tokenizer_kwargs = {}
    if args.model_revision:
        tokenizer_kwargs["revision"] = args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
    chat_template_active = bool(
        args.use_chat_template
        and getattr(tokenizer, "chat_template", None)
        and callable(getattr(tokenizer, "apply_chat_template", None))
    )
    add_special_tokens = not chat_template_active

    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        prompt = format_with_optional_chat_template(
            tokenizer,
            record["question"],
            answer=None,
            use_chat_template=args.use_chat_template,
        )
        full_text = format_with_optional_chat_template(
            tokenizer,
            record["question"],
            answer=record["answer"],
            use_chat_template=args.use_chat_template,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=add_special_tokens)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=add_special_tokens)["input_ids"]
        truncated_ids = tokenizer(
            full_text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=args.max_seq_length,
        )["input_ids"]
        answer_ids = tokenizer(record["answer"], add_special_tokens=False)["input_ids"]
        preview_ids = truncated_ids[: args.inspect_token_preview]
        tail_ids = truncated_ids[-args.inspect_token_preview :]
        rows.append(
            {
                "index": index,
                "question": record["question"],
                "answer": record["answer"],
                "chat_template_active": chat_template_active,
                "prompt_token_count": len(prompt_ids),
                "answer_token_count": len(answer_ids),
                "full_token_count": len(full_ids),
                "truncated_token_count": len(truncated_ids),
                "truncated": len(full_ids) > len(truncated_ids),
                "max_seq_length": args.max_seq_length,
                "first_tokens": tokenizer.convert_ids_to_tokens(preview_ids),
                "last_tokens": tokenizer.convert_ids_to_tokens(tail_ids),
                "decoded_tail": tokenizer.decode(tail_ids),
            }
        )

    write_json(
        args.tokenized_batch_output,
        {
            "model_name": args.model_name,
            "model_provenance": model_provenance(args),
            "use_chat_template": args.use_chat_template,
            "label_note": (
                "The SFTTrainer receives the full text field. Unless a completion-only "
                "data collator is added, non-padding labels include prompt and answer tokens."
            ),
            "records": rows,
        },
    )
    print(f"Wrote tokenization inspection to {args.tokenized_batch_output}")


def load_adapter_model_and_tokenizer(args: argparse.Namespace) -> tuple[object, object, object, Path]:
    """Load the base model and attach a saved PEFT adapter from disk."""

    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    adapter_dir = args.adapter_dir or (args.output_dir / "lora_adapter")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    model, tokenizer = load_model_and_tokenizer(args)
    model = PeftModel.from_pretrained(model, adapter_dir)
    FastLanguageModel.for_inference(model)
    return model, tokenizer, torch, adapter_dir


def run_adapter_evaluation(args: argparse.Namespace) -> None:
    """Reload a saved adapter and generate held-out validation answers."""

    missing = [name for name, present in dependency_status().items() if not present]
    if missing:
        raise SystemExit(
            "Missing adapter-evaluation dependencies: "
            + ", ".join(missing)
            + ". Install Unsloth, PEFT, and related packages in a compatible environment."
        )

    if args.adapter_output is None:
        args.adapter_output = args.output_dir / "adapter_outputs.jsonl"

    if args.adapter_output.parent:
        args.adapter_output.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(args.validation_data)
    started = time.perf_counter()
    model, tokenizer, torch_module, adapter_dir = load_adapter_model_and_tokenizer(args)

    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        prompt = format_with_optional_chat_template(
            tokenizer,
            record["question"],
            answer=None,
            use_chat_template=args.use_chat_template,
        )
        generated = generate_one(model, tokenizer, prompt, args, torch_module)
        rows.append(
            {
                "index": index,
                "question": record["question"],
                "reference_answer": record.get("answer", ""),
                "generated_answer": generated,
            }
        )

    elapsed = time.perf_counter() - started
    write_jsonl(args.adapter_output, rows)
    write_json(
        args.output_dir / f"{args.run_name}_adapter_evaluation_metadata.json",
        {
            "run_kind": "adapter_evaluation",
            "run_name": args.run_name,
            "model_name": args.model_name,
            "model_provenance": model_provenance(args),
            "adapter_dir": str(adapter_dir),
            "records": len(rows),
            "max_seq_length": args.max_seq_length,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "use_chat_template": args.use_chat_template,
            "wall_clock_seconds": elapsed,
            "dependencies": dependency_versions(),
            "gpu_memory": cuda_memory_summary(torch_module),
            "output_file": str(args.adapter_output),
        },
    )
    print(f"Wrote adapter generations to {args.adapter_output}")


def run_training(args: argparse.Namespace) -> None:
    """Run a small Unsloth SFT job.

    This function imports optional dependencies lazily so normal repository
    checks do not require a GPU environment.
    """

    missing = [name for name, present in dependency_status().items() if not present]
    if missing:
        raise SystemExit(
            "Missing training dependencies: "
            + ", ".join(missing)
            + ". Install Unsloth and related packages in a compatible environment."
        )

    import torch
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    records = load_records(args.data)
    started = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(args)
    dataset = Dataset.from_list(records_to_text(records, tokenizer, args.use_chat_template))

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    trainable_params = trainable_parameter_count(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=TrainingArguments(
            output_dir=str(args.output_dir),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            report_to=[],
        ),
    )
    trainer.train()
    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir / "lora_adapter")
    tokenizer.save_pretrained(args.output_dir / "lora_adapter")
    write_json(
        args.output_dir / f"{args.run_name}_{args.run_kind}_metadata.json",
        {
            "run_kind": args.run_kind,
            "run_name": args.run_name,
            "model_name": args.model_name,
            "model_provenance": model_provenance(args),
            "training_records": len(records),
            "max_seq_length": args.max_seq_length,
            "load_in_4bit": args.load_in_4bit,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
            "warmup_steps": args.warmup_steps,
            "seed": args.seed,
            "use_chat_template": args.use_chat_template,
            "trainable_parameters": trainable_params,
            "wall_clock_seconds": elapsed,
            "dependencies": dependency_versions(),
            "gpu_memory": cuda_memory_summary(torch),
            "adapter_dir": str(args.output_dir / "lora_adapter"),
        },
    )
    print(f"Saved LoRA adapter and metadata under {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--allow-missing-deps", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate-adapter", action="store_true")
    parser.add_argument("--inspect-tokenized-batch", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/unsloth_lora"))
    parser.add_argument("--baseline-output", type=Path, default=Path("runs/unsloth_lora/baseline_outputs.jsonl"))
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--adapter-output", type=Path)
    parser.add_argument(
        "--tokenized-batch-output",
        type=Path,
        default=Path("runs/unsloth_lora/tokenized_batch.json"),
    )
    parser.add_argument("--inspect-examples", type=int, default=3)
    parser.add_argument("--inspect-token-preview", type=int, default=24)
    parser.add_argument("--run-name", default="course_assistant")
    parser.add_argument("--run-kind", choices=["lora", "ablation"], default="lora")
    parser.add_argument("--model-name", default="unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--model-license", default="")
    parser.add_argument("--model-card-url", default="")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.check_deps:
        status = dependency_status()
        print(json.dumps({"available": status, "versions": dependency_versions()}, indent=2, sort_keys=True))
        if not args.allow_missing_deps and not all(status.values()):
            missing = [name for name, present in status.items() if not present]
            raise SystemExit("Missing dependencies: " + ", ".join(missing))
        return

    if args.baseline:
        run_prompt_only_baseline(args)
        return

    if args.inspect_tokenized_batch:
        inspect_tokenized_batch(args)
        return

    if args.evaluate_adapter:
        run_adapter_evaluation(args)
        return

    if not args.train:
        preview = [format_prompt(record["question"], record["answer"]) for record in built_in_records()]
        print(json.dumps({"preview": preview}, indent=2))
        print("Use --baseline for prompt-only outputs, --train for LoRA training, or --evaluate-adapter to reload a saved adapter.")
        return

    run_training(args)


if __name__ == "__main__":
    main()
