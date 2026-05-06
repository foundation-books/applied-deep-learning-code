#!/usr/bin/env python3
"""Small Qwen3-TTS voice-cloning workflow scaffold.

The script keeps the chapter code testable without requiring model downloads.
It can create sample manifests, compute simple WER/CER from text hypotheses,
check optional dependencies, and optionally call Qwen3-TTS when the runtime
environment provides the required package and model access.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence


OPTIONAL_DEPENDENCIES = {
    "qwen_tts": "qwen-tts",
    "soundfile": "soundfile",
    "torch": "torch",
}

DEFAULT_MANIFEST_PATH = Path(__file__).with_name("tts_eval_manifest.csv")
RESULT_FIELDNAMES = [
    "target_id",
    "target_text",
    "hypothesis_text",
    "generated_audio",
    "generated_seconds",
    "wall_time_seconds",
    "rtf",
    "wer",
    "cer",
    "speaker_score",
    "naturalness_score",
    "notes",
]
SUMMARY_FIELDNAMES = [
    "model_id",
    "mode",
    "ref_audio_seconds",
    "language",
    "target_count",
    "total_audio_seconds",
    "wall_time_seconds",
    "rtf",
    "wer",
    "cer",
    "speaker_similarity",
    "naturalness",
    "notes",
]


SAMPLE_TARGETS = [
    ("target_001", "A waveform is a sequence of audio samples.", "English", "clear classroom narration"),
    ("target_002", "Text to speech maps written text and voice conditions to audio.", "English", "clear classroom narration"),
    ("target_003", "Real-time factor compares runtime with generated audio duration.", "English", "clear classroom narration"),
    ("target_004", "The model predicts speech tokens before the waveform is reconstructed.", "English", "clear classroom narration"),
    ("target_005", "A clean reference recording makes voice cloning easier to evaluate.", "English", "clear classroom narration"),
    ("target_006", "The target manifest should stay fixed during a controlled comparison.", "English", "clear classroom narration"),
    ("target_007", "Dr. Lee measured 3.14 kilograms in the laboratory.", "English", "careful pronunciation"),
    ("target_008", "ASR backcheck can reveal skipped words in generated speech.", "English", "careful pronunciation"),
]


def normalize_text(text: str) -> str:
    """Normalize text for simple classroom WER/CER examples."""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Compute Levenshtein edit distance between two token sequences."""

    prev = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        curr = [i]
        for j, item_b in enumerate(b, start=1):
            cost = 0 if item_a == item_b else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def check_dependencies(allow_missing: bool) -> int:
    missing = []
    for module_name, package_name in OPTIONAL_DEPENDENCIES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)

    if missing:
        message = "Missing optional dependencies: " + ", ".join(missing)
        if allow_missing:
            print(message)
            print("Continuing because --allow-missing-deps was provided.")
            return 0
        print(message, file=sys.stderr)
        return 1

    print("All optional Qwen3-TTS dependencies are importable.")
    return 0


def package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_environment() -> dict[str, object]:
    return {
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python_version": sys.version.split()[0],
        "package_versions": {
            package_name: package_version(package_name)
            for package_name in OPTIONAL_DEPENDENCIES.values()
        },
    }


def validate_homework_target_count(targets: Sequence[object]) -> None:
    if not 8 <= len(targets) <= 12:
        raise ValueError(
            "The chapter homework expects 8 to 12 target sentences; "
            f"found {len(targets)}."
        )


def load_sample_targets() -> list[dict[str, str]]:
    if DEFAULT_MANIFEST_PATH.exists():
        rows = read_csv_rows(DEFAULT_MANIFEST_PATH)
        return [
            {
                "target_id": row["target_id"],
                "text": row["text"],
                "language": row.get("language", ""),
                "style": row.get("style", ""),
            }
            for row in rows
        ]

    return [
        {
            "target_id": target_id,
            "text": text,
            "language": language,
            "style": style,
        }
        for target_id, text, language, style in SAMPLE_TARGETS
    ]


def write_sample_data(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_targets = load_sample_targets()

    (output_dir / "reference.txt").write_text(
        "This is my reference recording for the text to speech chapter example.\n",
        encoding="utf-8",
    )

    with (output_dir / "targets.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["target_id", "text", "language", "style"])
        writer.writeheader()
        for target in sample_targets:
            writer.writerow(
                {
                    "target_id": target["target_id"],
                    "text": target["text"],
                    "language": target["language"],
                    "style": target["style"],
                }
            )

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for target in sample_targets:
            writer.writerow(
                {
                    "target_id": target["target_id"],
                    "target_text": target["text"],
                    "hypothesis_text": "",
                    "generated_audio": "",
                    "generated_seconds": "",
                    "wall_time_seconds": "",
                    "rtf": "",
                    "wer": "",
                    "cer": "",
                    "speaker_score": "",
                    "naturalness_score": "",
                    "notes": "",
                }
            )

    with (output_dir / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                field: f"[RESULT: {field}]"
                for field in SUMMARY_FIELDNAMES
            }
        )

    metadata = {
        "chapter": "text-to-speech",
        "task": "Qwen3-TTS voice cloning",
        "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "mode": "voice_clone",
        "reference_audio": "reference.wav",
        "reference_text": "reference.txt",
        "reference_audio_seconds": "[RESULT: ref_audio_seconds]",
        "targets": "targets.csv",
        "target_count": len(sample_targets),
        "results": "results.csv",
        "experiment_summary": "experiment_summary.csv",
        "runtime_backend": "[RESULT: runtime_backend]",
        "dtype": "bfloat16",
        "generation_settings": {
            "language": "English",
            "style": "per target manifest",
        },
        "result_placeholders": [
            "model_id",
            "mode",
            "ref_audio_seconds",
            "language",
            "target_count",
            "total_audio_seconds",
            "wall_time_seconds",
            "rtf",
            "wer",
            "cer",
            "speaker_similarity",
            "naturalness",
            "notes",
            "generated_seconds",
            "speaker_score",
            "naturalness_score",
        ],
        **runtime_environment(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote sample TTS homework files to {output_dir}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def score_results(results_csv: Path) -> None:
    rows = read_csv_rows(results_csv)
    scored_rows = []
    for row in rows:
        reference = row.get("target_text", "")
        hypothesis = row.get("hypothesis_text", "")
        generated_seconds = safe_float(row.get("generated_seconds", ""))
        wall_time_seconds = safe_float(row.get("wall_time_seconds", ""))
        rtf = None
        if generated_seconds and generated_seconds > 0 and wall_time_seconds is not None:
            rtf = wall_time_seconds / generated_seconds

        row_score = {
            "target_id": row.get("target_id", ""),
            "wer": None if not hypothesis else wer(reference, hypothesis),
            "cer": None if not hypothesis else cer(reference, hypothesis),
            "rtf": rtf,
            "speaker_score": safe_float(row.get("speaker_score", "")),
            "naturalness_score": safe_float(row.get("naturalness_score", "")),
            "notes": row.get("notes", ""),
        }
        scored_rows.append(row_score)

    print(json.dumps({"rows": scored_rows}, indent=2))


def iter_targets(path: Path) -> Iterable[dict[str, str]]:
    rows = read_csv_rows(path)
    for i, row in enumerate(rows, start=1):
        target_id = row.get("target_id") or f"target_{i:03d}"
        text = row.get("text") or row.get("target_text")
        if not text:
            raise ValueError(f"Missing text for row {i} in {path}")
        yield {
            "target_id": target_id,
            "text": text,
            "language": row.get("language", ""),
            "style": row.get("style", ""),
        }


def run_qwen(args: argparse.Namespace) -> None:
    missing_status = check_dependencies(allow_missing=False)
    if missing_status != 0:
        raise SystemExit(missing_status)

    import soundfile as sf  # type: ignore
    import torch  # type: ignore
    from qwen_tts import Qwen3TTSModel  # type: ignore

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref_text = args.reference_text.read_text(encoding="utf-8").strip()
    target_rows = list(iter_targets(args.targets))
    validate_homework_target_count(target_rows)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = Qwen3TTSModel.from_pretrained(
        args.model_id,
        device_map=args.device,
        dtype=dtype,
    )

    result_rows = []
    total_audio_seconds = 0.0
    total_start = time.perf_counter()
    for target in target_rows:
        target_id = target["target_id"]
        text = target["text"]
        language = target["language"] or args.language
        style = target["style"] or args.style
        start = time.perf_counter()
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=str(args.reference_audio),
            ref_text=ref_text,
        )
        wall_time = time.perf_counter() - start
        output_path = args.output_dir / f"{target_id}.wav"
        sf.write(output_path, wavs[0], sample_rate)
        generated_seconds = len(wavs[0]) / float(sample_rate)
        total_audio_seconds += generated_seconds
        result_rows.append(
            {
                "target_id": target_id,
                "target_text": text,
                "hypothesis_text": "",
                "generated_audio": str(output_path),
                "generated_seconds": f"{generated_seconds:.3f}",
                "wall_time_seconds": f"{wall_time:.3f}",
                "rtf": f"{wall_time / generated_seconds:.3f}" if generated_seconds else "",
                "wer": "",
                "cer": "",
                "speaker_score": "",
                "naturalness_score": "",
                "notes": f"style={style}" if style else "",
            }
        )
    total_wall_time = time.perf_counter() - total_start

    results_path = args.output_dir / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(result_rows)

    try:
        reference_audio_seconds = float(sf.info(str(args.reference_audio)).duration)
    except RuntimeError:
        reference_audio_seconds = None

    summary = {
        "model_id": args.model_id,
        "mode": "voice_clone",
        "ref_audio_seconds": "" if reference_audio_seconds is None else f"{reference_audio_seconds:.3f}",
        "language": args.language,
        "target_count": str(len(target_rows)),
        "total_audio_seconds": f"{total_audio_seconds:.3f}",
        "wall_time_seconds": f"{total_wall_time:.3f}",
        "rtf": f"{total_wall_time / total_audio_seconds:.3f}" if total_audio_seconds else "",
        "wer": "",
        "cer": "",
        "speaker_similarity": "",
        "naturalness": "",
        "notes": "Fill WER/CER after ASR-backcheck and listener scores after evaluation.",
    }
    with (args.output_dir / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerow(summary)

    metadata = {
        "model_id": args.model_id,
        "mode": "voice_clone",
        "device": args.device,
        "dtype": args.dtype,
        "runtime_backend": "local_qwen_tts",
        "reference_audio": str(args.reference_audio),
        "reference_text": str(args.reference_text),
        "reference_audio_seconds": reference_audio_seconds,
        "targets": str(args.targets),
        "target_count": len(target_rows),
        "output_dir": str(args.output_dir),
        "language": args.language,
        "style": args.style,
        "public_release_note": (
            "Do not publish reference voice audio, copied transcripts, or generated "
            "voice-clone audio unless you have the needed consent, attribution, "
            "and model-provider rights for that release."
        ),
        "generation_settings": {
            "model_id": args.model_id,
            "language_default": args.language,
            "style_default": args.style,
            "device": args.device,
            "dtype": args.dtype,
        },
        "total_audio_seconds": total_audio_seconds,
        "total_wall_time_seconds": total_wall_time,
        "aggregate_rtf": total_wall_time / total_audio_seconds if total_audio_seconds else None,
        **runtime_environment(),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Qwen3-TTS outputs and results to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-deps", action="store_true", help="Check optional Qwen3-TTS dependencies.")
    parser.add_argument("--allow-missing-deps", action="store_true", help="Allow dependency check to pass when optional packages are missing.")
    parser.add_argument("--write-sample-data", type=Path, help="Write sample targets and result templates to this directory.")
    parser.add_argument("--score-results", type=Path, help="Score a completed or partially completed results CSV.")
    parser.add_argument("--run-qwen", action="store_true", help="Run local Qwen3-TTS voice cloning.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--reference-text", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--language", default="English")
    parser.add_argument("--style", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.check_deps:
        return check_dependencies(args.allow_missing_deps)

    if args.write_sample_data:
        write_sample_data(args.write_sample_data)
        return 0

    if args.score_results:
        score_results(args.score_results)
        return 0

    if args.run_qwen:
        required = {
            "--reference-audio": args.reference_audio,
            "--reference-text": args.reference_text,
            "--targets": args.targets,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("--run-qwen requires " + ", ".join(missing))
        run_qwen(args)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
