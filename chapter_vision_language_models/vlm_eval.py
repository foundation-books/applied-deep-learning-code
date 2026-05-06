#!/usr/bin/env python3
"""VLM evaluation scaffold for the visual math homework.

The repository-safe path tests data format, parser behavior, and scoring without
requiring a GPU or model downloads. A real prompt-only VLM run is available with
--generate when PyTorch, Transformers, Pillow, and a compatible checkpoint are
installed.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Iterable

from answer_parser import score_completion, unit_test_parser


OPTIONAL_RUNTIME_PACKAGES = {
    "accelerate": "accelerate",
    "pillow": "PIL",
    "torch": "torch",
    "transformers": "transformers",
}
SAMPLE_PROMPTS_PATH = Path(__file__).with_name("sample_visual_math_prompts.jsonl")
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
ANSWER_CONTRACT = (
    "Read the image and answer the question. Put exactly one final answer inside "
    "<answer>...</answer> tags. Keep the answer short."
)


def package_available(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def dependency_status() -> dict[str, bool]:
    return {
        package_name: package_available(import_name)
        for package_name, import_name in OPTIONAL_RUNTIME_PACKAGES.items()
    }


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in OPTIONAL_RUNTIME_PACKAGES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if "question" not in record:
                raise ValueError(f"{path}:{line_number} missing question")
            if "reference_answer" not in record and "answer" not in record:
                raise ValueError(f"{path}:{line_number} missing reference_answer")
            normalized = dict(record)
            normalized.setdefault("id", f"row_{line_number:03d}")
            normalized.setdefault("split", "unspecified")
            if "reference_answer" not in normalized:
                normalized["reference_answer"] = normalized["answer"]
            records.append(normalized)
    return records


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_records(path: Path | None) -> list[dict[str, object]]:
    return read_jsonl(path or SAMPLE_PROMPTS_PATH)


def prompt_for_question(question: str, answer_contract: str = ANSWER_CONTRACT) -> str:
    return f"{answer_contract}\n\nQuestion: {question}"


def score_generated_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    scored: list[dict[str, object]] = []
    for row in rows:
        completion = str(
            row.get("generated_text")
            or row.get("completion")
            or row.get("model_output")
            or ""
        )
        reference = str(row.get("reference_answer") or row.get("final_answer") or row.get("answer") or "")
        score = score_completion(completion, reference)
        scored.append({**row, "generated_text": completion, **score.to_dict()})
    return scored


def summarize_scores(rows: list[dict[str, object]]) -> dict[str, object]:
    count = len(rows)
    exact = sum(1 for row in rows if row.get("exact_match"))
    valid = sum(1 for row in rows if row.get("valid_format"))
    parser_errors: dict[str, int] = {}
    error_taxonomy: dict[str, int] = {}
    latencies = []
    throughputs = []

    for row in rows:
        parser_error = str(row.get("parser_error") or "none")
        parser_errors[parser_error] = parser_errors.get(parser_error, 0) + 1
        if not row.get("exact_match"):
            category = str(row.get("error_category") or parser_error)
            error_taxonomy[category] = error_taxonomy.get(category, 0) + 1
        if isinstance(row.get("first_token_seconds"), int | float):
            latencies.append(float(row["first_token_seconds"]))
        if isinstance(row.get("output_tokens_per_second"), int | float):
            throughputs.append(float(row["output_tokens_per_second"]))

    split_rows: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        split = str(row.get("split") or "unspecified")
        split_rows.setdefault(split, []).append(row)

    return {
        "count": count,
        "exact_match_count": exact,
        "exact_match_rate": exact / count if count else None,
        "valid_format_count": valid,
        "valid_format_rate": valid / count if count else None,
        "parser_errors": parser_errors,
        "error_taxonomy": error_taxonomy,
        "mean_first_token_seconds": sum(latencies) / len(latencies) if latencies else None,
        "mean_output_tokens_per_second": (
            sum(throughputs) / len(throughputs) if throughputs else None
        ),
        "by_split": {
            split: {
                "count": len(split_group),
                "exact_match_count": sum(1 for row in split_group if row.get("exact_match")),
                "exact_match_rate": (
                    sum(1 for row in split_group if row.get("exact_match")) / len(split_group)
                ),
                "valid_format_count": sum(1 for row in split_group if row.get("valid_format")),
                "valid_format_rate": (
                    sum(1 for row in split_group if row.get("valid_format")) / len(split_group)
                ),
            }
            for split, split_group in sorted(split_rows.items())
            if split_group
        },
    }


FONT_5X7 = {
    "0": ("11111", "10001", "10011", "10101", "11001", "10001", "11111"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("11110", "00001", "00001", "11110", "10000", "10000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("10010", "10010", "10010", "11111", "00010", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01111", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "11110"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "X": ("10001", "01010", "00100", "00100", "00100", "01010", "10001"),
    "=": ("00000", "11111", "00000", "00000", "11111", "00000", "00000"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
}


class Canvas:
    def __init__(self, width: int = 240, height: int = 140) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray([255, 255, 255] * width * height)

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            index = (y * self.width + x) * 3
            self.pixels[index:index + 3] = bytes(color)

    def fill_rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        color: tuple[int, int, int],
    ) -> None:
        for yy in range(y, y + height):
            for xx in range(x, x + width):
                self.set_pixel(xx, yy, color)

    def draw_text(
        self,
        text: str,
        x: int,
        y: int,
        scale: int = 5,
        color: tuple[int, int, int] = (20, 20, 20),
    ) -> None:
        cursor = x
        for char in text.upper():
            if char == " ":
                cursor += 4 * scale
                continue
            glyph = FONT_5X7.get(char)
            if glyph is None:
                cursor += 6 * scale
                continue
            for row_index, row in enumerate(glyph):
                for col_index, value in enumerate(row):
                    if value == "1":
                        self.fill_rect(
                            cursor + col_index * scale,
                            y + row_index * scale,
                            scale,
                            scale,
                            color,
                        )
            cursor += 6 * scale

    def write_png(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = bytearray()
        row_width = self.width * 3
        for y in range(self.height):
            raw.append(0)
            start = y * row_width
            raw.extend(self.pixels[start:start + row_width])
        png_signature = b"\x89PNG\r\n\x1a\n"
        payload = b"".join(
            [
                _png_chunk(
                    b"IHDR",
                    struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0),
                ),
                _png_chunk(b"IDAT", zlib.compress(bytes(raw))),
                _png_chunk(b"IEND", b""),
            ]
        )
        path.write_bytes(png_signature + payload)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _draw_expression(path: Path, expression: str) -> None:
    canvas = Canvas()
    text_width = len(expression.replace(" ", "")) * 30
    canvas.draw_text(expression, max(12, (canvas.width - text_width) // 2), 48, scale=6)
    canvas.write_png(path)


def _draw_bar_chart(path: Path, values: tuple[int, int, int] = (4, 8, 6)) -> None:
    canvas = Canvas()
    axis = (20, 20, 20)
    canvas.fill_rect(35, 18, 3, 92, axis)
    canvas.fill_rect(35, 108, 170, 3, axis)
    bars = [
        ("A", values[0], (75, 130, 190)),
        ("B", values[1], (230, 150, 60)),
        ("C", values[2], (90, 160, 95)),
    ]
    for index, (label, value, color) in enumerate(bars):
        x = 58 + index * 48
        height = value * 10
        canvas.fill_rect(x, 108 - height, 28, height, color)
        canvas.draw_text(label, x + 2, 116, scale=3)
        canvas.draw_text(str(value), x + 6, 92 - height, scale=3)
    canvas.write_png(path)


def sample_records(output_dir: Path) -> list[dict[str, object]]:
    expressions = [
        ("7+5", 12),
        ("9-4", 5),
        ("3X6", 18),
        ("8+8", 16),
        ("6+7", 13),
        ("9X2", 18),
        ("15-7", 8),
        ("4X4", 16),
        ("12+6", 18),
        ("18-9", 9),
        ("5X5", 25),
        ("14+3", 17),
        ("20-8", 12),
        ("2X9", 18),
        ("11+8", 19),
        ("16-6", 10),
        ("7X3", 21),
        ("13+4", 17),
        ("19-5", 14),
        ("6X6", 36),
        ("10+9", 19),
        ("17-8", 9),
        ("8X3", 24),
        ("12-5", 7),
    ]
    chart_values = [
        ((4, 8, 6), "B"),
        ((9, 5, 7), "A"),
        ((3, 6, 10), "C"),
        ((8, 9, 4), "B"),
        ((7, 2, 5), "A"),
        ((5, 6, 8), "C"),
    ]
    splits = ["train"] * 12 + ["validation"] * 9 + ["final_test"] * 9
    records: list[dict[str, object]] = []

    for index, (expression, answer) in enumerate(expressions, start=1):
        record_index = len(records)
        image_name = f"expression_{index:03d}.png"
        records.append(
            {
                "id": f"expression_{index:03d}",
                "split": splits[record_index],
                "image_path": str(output_dir / image_name),
                "question": "Read the arithmetic expression in the image. What is its value?",
                "reference_answer": str(answer),
                "image_description": f"The image shows {expression.replace('X', ' x ')}.",
            }
        )

    for index, (values, answer) in enumerate(chart_values, start=1):
        record_index = len(records)
        image_name = f"chart_{index:03d}.png"
        records.append(
            {
                "id": f"chart_{index:03d}",
                "split": splits[record_index],
                "image_path": str(output_dir / image_name),
                "question": "Which bar is tallest: A, B, or C?",
                "reference_answer": answer,
                "image_description": (
                    f"A small bar chart has values A={values[0]}, "
                    f"B={values[1]}, and C={values[2]}."
                ),
            }
        )

    return records


def write_sample_data(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = sample_records(output_dir)
    for record in records:
        image_path = Path(str(record["image_path"]))
        if str(record["id"]).startswith("expression_"):
            expression = str(record["image_description"]).removeprefix("The image shows ")
            expression = expression.removesuffix(".").replace(" x ", "X").replace(" ", "")
            _draw_expression(image_path, expression)
        elif str(record["id"]).startswith("chart_"):
            description = str(record["image_description"])
            values = tuple(
                int(part.split("=")[1].strip(" ."))
                for part in description.replace("and ", "").split("values ", 1)[1].split(", ")
            )
            _draw_bar_chart(image_path, values)  # type: ignore[arg-type]
    prompts_path = output_dir / "prompts.jsonl"
    write_jsonl(prompts_path, records)
    return prompts_path


def preview(args: argparse.Namespace) -> None:
    records = load_records(args.prompts)
    for record in records[: args.limit]:
        print(f"[{record['id']}] split={record['split']}")
        print(f"image: {record.get('image_path', '<missing>')}")
        print(prompt_for_question(str(record["question"]), args.answer_contract))
        print(f"reference: {record['reference_answer']}")
        print()


def score_jsonl(path: Path, output: Path, summary_output: Path) -> None:
    rows = read_jsonl(path)
    scored = score_generated_rows(rows)
    write_jsonl(output, scored)
    write_json(summary_output, summarize_scores(scored))
    print(f"wrote scored rows to {output}")
    print(f"wrote summary to {summary_output}")


def metadata_template() -> dict[str, object]:
    return {
        "contract": {
            "answer_format": "exactly one <answer>...</answer> tag",
            "exact_match_normalization": "case, whitespace, simple decimal variants",
            "valid_format_rule": "one non-empty answer tag",
        },
        "data": {
            "prompt_path": "data/sample_visual_math/prompts.jsonl",
            "split_rule": "train for adaptation, validation for choices, final_test once",
            "minimum_homework_examples": 30,
        },
        "model": {
            "checkpoint": DEFAULT_MODEL_NAME,
            "image_size_or_processor": None,
            "prompt_template": ANSWER_CONTRACT,
            "max_new_tokens": 64,
            "temperature": 0.0,
        },
        "adaptation_or_ablation": {
            "method": "LoRA SFT, VLM RL/GRPO, or controlled fallback ablation",
            "changed_factor": None,
            "lora_rank": None,
            "target_modules": None,
            "reward_weights": {"correctness": 1.0, "format": 0.2},
        },
        "measurements": {
            "exact_match_rate": None,
            "valid_format_rate": None,
            "first_token_seconds": None,
            "output_tokens_per_second": None,
            "peak_memory": None,
            "error_taxonomy": None,
        },
    }


def write_metadata_template(path: Path) -> None:
    write_json(path, metadata_template())
    print(f"wrote metadata template to {path}")


def load_vlm_runtime() -> tuple[object, object, object, object]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    transformers = importlib.import_module("transformers")
    model_class = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_class is None:
        model_class = getattr(transformers, "AutoModelForVision2Seq", None)
    if model_class is None:
        raise RuntimeError(
            "This Transformers version does not expose AutoModelForImageTextToText "
            "or AutoModelForVision2Seq."
        )
    return torch, Image, AutoProcessor, model_class


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


def apply_chat_template(processor: object, question: str, answer_contract: str) -> str:
    prompt = prompt_for_question(question, answer_contract)
    apply_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_template):
        return prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return str(
        apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def decode_generated(processor: object, generated_ids: object) -> str:
    if hasattr(processor, "batch_decode"):
        return str(
            processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        ).strip()
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor has neither batch_decode nor tokenizer.")
    return str(tokenizer.decode(generated_ids[0], skip_special_tokens=True)).strip()


def move_inputs_to_device(inputs: object, device: object) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in dict(inputs).items():
        moved[key] = value.to(device) if hasattr(value, "to") and device is not None else value
    return moved


def generate_baseline(args: argparse.Namespace) -> None:
    missing = [name for name, available in dependency_status().items() if not available]
    if missing:
        raise SystemExit(
            "Missing optional runtime packages for --generate: " + ", ".join(missing)
        )

    records = load_records(args.prompts)
    torch, Image, AutoProcessor, AutoModel = load_vlm_runtime()
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        device_map=args.device_map,
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    outputs: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        image_path = Path(str(record.get("image_path", ""))).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(f"{record['id']} image_path does not exist: {image_path}")

        image = Image.open(image_path).convert("RGB")
        prompt_text = apply_chat_template(
            processor,
            str(record["question"]),
            args.answer_contract,
        )
        inputs = processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        device = getattr(model, "device", None)
        inputs = move_inputs_to_device(inputs, device)
        prompt_tokens = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else None

        first_token_seconds = None
        if not args.skip_first_token_timing:
            first_started = time.perf_counter()
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                )
            first_token_seconds = time.perf_counter() - first_started

        generation_started = time.perf_counter()
        generation_kwargs: dict[str, object] = {
            **inputs,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p
        with torch.inference_mode():
            generated = model.generate(**generation_kwargs)
        generation_seconds = time.perf_counter() - generation_started

        if prompt_tokens is not None and generated.shape[-1] > prompt_tokens:
            completion_ids = generated[:, prompt_tokens:]
        else:
            completion_ids = generated
        generated_text = decode_generated(processor, completion_ids)
        new_tokens = int(completion_ids.shape[-1])
        score = score_completion(generated_text, str(record["reference_answer"]))
        outputs.append(
            {
                **record,
                "index": index,
                "prompt": prompt_text,
                "model_name": args.model_name,
                "prompt_token_count": prompt_tokens,
                "generated_token_count": new_tokens,
                "first_token_seconds": first_token_seconds,
                "full_generation_seconds": generation_seconds,
                "output_tokens_per_second": (
                    new_tokens / generation_seconds if generation_seconds else None
                ),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "generated_text": generated_text,
                **score.to_dict(),
            }
        )

    write_jsonl(args.output, outputs)
    write_json(args.summary_output, summarize_scores(outputs))
    write_json(
        args.metadata_output,
        {
            "model_name": args.model_name,
            "prompt_count": len(records),
            "answer_contract": args.answer_contract,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "wall_clock_seconds": time.perf_counter() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "packages": dependency_versions(),
            "cuda": cuda_summary(torch),
        },
    )
    print(f"wrote generations to {args.output}")
    print(f"wrote summary to {args.summary_output}")
    print(f"wrote metadata to {args.metadata_output}")


def print_training_note() -> None:
    print(
        "For adaptation, keep this scaffold as the scoring harness. Run LoRA SFT "
        "or VLM RL/GRPO in a current GPU notebook, save model outputs as JSONL with "
        "generated_text and reference_answer fields, then score them with --score-jsonl."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--allow-missing-deps", action="store_true")
    parser.add_argument("--unit-test-parser", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--write-sample-data", type=Path)
    parser.add_argument("--write-metadata-template", type=Path)
    parser.add_argument("--score-jsonl", type=Path)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--training-note", action="store_true")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--answer-contract", default=ANSWER_CONTRACT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--skip-first-token-timing", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/vlm_prompt_outputs.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("runs/vlm_prompt_summary.json"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("runs/vlm_prompt_metadata.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    handled = False

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
            print("Missing optional runtime packages: " + ", ".join(missing))
            return 1
        handled = True

    if args.unit_test_parser:
        unit_test_parser()
        print("parser unit tests passed")
        handled = True

    if args.write_sample_data:
        prompts_path = write_sample_data(args.write_sample_data)
        print(f"wrote sample prompts and images under {args.write_sample_data}")
        print(f"prompt file: {prompts_path}")
        handled = True

    if args.write_metadata_template:
        write_metadata_template(args.write_metadata_template)
        handled = True

    if args.preview:
        preview(args)
        handled = True

    if args.score_jsonl:
        score_jsonl(args.score_jsonl, args.output, args.summary_output)
        handled = True

    if args.generate:
        generate_baseline(args)
        handled = True

    if args.training_note:
        print_training_note()
        handled = True

    if not handled:
        print("Use --help for parser tests, sample data, scoring, and generation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
