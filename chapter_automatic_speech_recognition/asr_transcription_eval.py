#!/usr/bin/env python3
"""Dependency-light ASR homework scoring scaffold.

The scaffold validates an audio manifest, computes WER and CER for saved ASR
hypotheses, measures real-time factor from recorded processing time, and writes
artifacts for the automatic speech recognition chapter homework. It deliberately
does not download a pretrained recognizer during repository checks.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import statistics
import struct
import sys
import time
from typing import Iterable
import wave


TOKEN_RE = re.compile(r"[a-z0-9']+")
OPTIONAL_PACKAGES = ["qwen_asr", "torch", "transformers", "soundfile", "jiwer"]
ERROR_CLASSES = {
    "acoustic",
    "language",
    "domain_vocabulary",
    "boundary_timestamp",
    "hallucination",
    "decoding_prompt",
    "other",
}


@dataclass(frozen=True)
class ManifestRecord:
    audio_id: str
    path: Path
    split: str
    language: str
    condition: str
    reference_text: str
    baseline_hypothesis: str
    controlled_hypothesis: str
    baseline_processing_seconds: float
    controlled_processing_seconds: float
    error_class: str


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_status() -> dict[str, bool]:
    return {name: package_available(name) for name in OPTIONAL_PACKAGES}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def normalize_text(text: str, keep_apostrophe: bool = True) -> str:
    lowered = text.lower()
    if keep_apostrophe:
        tokens = TOKEN_RE.findall(lowered)
    else:
        tokens = re.findall(r"[a-z0-9]+", lowered)
    return " ".join(tokens)


def word_tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def char_tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [ch for ch in normalized.replace(" ", "")]


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Return substitution, deletion, and insertion counts."""

    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)
    ]
    for i in range(1, rows):
        cost, s, d, ins = dp[i - 1][0]
        dp[i][0] = (cost + 1, s, d + 1, ins)
    for j in range(1, cols):
        cost, s, d, ins = dp[0][j - 1]
        dp[0][j] = (cost + 1, s, d, ins + 1)

    for i in range(1, rows):
        for j in range(1, cols):
            candidates: list[tuple[int, int, int, int]] = []
            cost, s, d, ins = dp[i - 1][j]
            candidates.append((cost + 1, s, d + 1, ins))
            cost, s, d, ins = dp[i][j - 1]
            candidates.append((cost + 1, s, d, ins + 1))
            cost, s, d, ins = dp[i - 1][j - 1]
            if reference[i - 1] == hypothesis[j - 1]:
                candidates.append((cost, s, d, ins))
            else:
                candidates.append((cost + 1, s + 1, d, ins))
            dp[i][j] = min(candidates, key=lambda item: (item[0], item[1] + item[2], item[3]))
    _, substitutions, deletions, insertions = dp[-1][-1]
    return substitutions, deletions, insertions


def error_rate(reference_units: list[str], hypothesis_units: list[str]) -> dict[str, object]:
    substitutions, deletions, insertions = edit_counts(reference_units, hypothesis_units)
    denominator = len(reference_units)
    rate = math.nan if denominator == 0 else (substitutions + deletions + insertions) / denominator
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_units": denominator,
        "rate": rate,
    }


def audio_duration_seconds(path: Path) -> tuple[float, int, int]:
    with wave.open(str(path), "rb") as handle:
        frame_count = handle.getnframes()
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
    return frame_count / sample_rate, sample_rate, channels


def resolve_audio_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def positive_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, 0.0)


def read_manifest(path: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            audio_id = str(record.get("audio_id", f"clip{line_number:03d}")).strip()
            raw_path = str(record.get("path", "")).strip()
            reference = str(record.get("reference_text", "")).strip()
            if not raw_path:
                raise ValueError(f"{path}:{line_number} missing path")
            if not reference:
                raise ValueError(f"{path}:{line_number} missing reference_text")
            error_class = str(record.get("error_class", "other")).strip() or "other"
            if error_class not in ERROR_CLASSES:
                error_class = "other"
            records.append(
                ManifestRecord(
                    audio_id=audio_id,
                    path=resolve_audio_path(path, raw_path),
                    split=str(record.get("split", "validation")),
                    language=str(record.get("language", "")),
                    condition=str(record.get("condition", "")),
                    reference_text=reference,
                    baseline_hypothesis=str(record.get("baseline_hypothesis", "")),
                    controlled_hypothesis=str(record.get("controlled_hypothesis", "")),
                    baseline_processing_seconds=positive_float(
                        record.get("baseline_processing_seconds"), default=0.0
                    ),
                    controlled_processing_seconds=positive_float(
                        record.get("controlled_processing_seconds"), default=0.0
                    ),
                    error_class=error_class,
                )
            )
    if not records:
        raise ValueError(f"No manifest records found in {path}")
    return records


def score_hypothesis(reference: str, hypothesis: str) -> dict[str, object]:
    wer = error_rate(word_tokens(reference), word_tokens(hypothesis))
    cer = error_rate(char_tokens(reference), char_tokens(hypothesis))
    return {
        "normalized_reference": normalize_text(reference),
        "normalized_hypothesis": normalize_text(hypothesis),
        "wer": wer,
        "cer": cer,
    }


def score_record(record: ManifestRecord) -> dict[str, object]:
    duration, sample_rate, channels = audio_duration_seconds(record.path)
    baseline = score_hypothesis(record.reference_text, record.baseline_hypothesis)
    controlled = score_hypothesis(record.reference_text, record.controlled_hypothesis)
    baseline_rtf = (
        record.baseline_processing_seconds / duration if duration > 0 else math.nan
    )
    controlled_rtf = (
        record.controlled_processing_seconds / duration if duration > 0 else math.nan
    )
    return {
        "audio_id": record.audio_id,
        "path": str(record.path),
        "split": record.split,
        "language": record.language,
        "condition": record.condition,
        "duration_seconds": duration,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "reference_text": record.reference_text,
        "baseline_hypothesis": record.baseline_hypothesis,
        "controlled_hypothesis": record.controlled_hypothesis,
        "baseline_processing_seconds": record.baseline_processing_seconds,
        "controlled_processing_seconds": record.controlled_processing_seconds,
        "baseline_rtf": baseline_rtf,
        "controlled_rtf": controlled_rtf,
        "baseline": baseline,
        "controlled": controlled,
        "error_class": record.error_class,
    }


def mean(values: Iterable[float]) -> float:
    usable = [value for value in values if not math.isnan(value)]
    return statistics.fmean(usable) if usable else math.nan


def summarize(scored_rows: list[dict[str, object]]) -> dict[str, object]:
    error_classes = Counter(str(row["error_class"]) for row in scored_rows)
    splits = Counter(str(row["split"]) for row in scored_rows)
    durations = [float(row["duration_seconds"]) for row in scored_rows]
    return {
        "clip_count": len(scored_rows),
        "split_counts": dict(sorted(splits.items())),
        "total_duration_seconds": sum(durations),
        "baseline_wer": mean(float(row["baseline"]["wer"]["rate"]) for row in scored_rows),  # type: ignore[index]
        "baseline_cer": mean(float(row["baseline"]["cer"]["rate"]) for row in scored_rows),  # type: ignore[index]
        "controlled_wer": mean(float(row["controlled"]["wer"]["rate"]) for row in scored_rows),  # type: ignore[index]
        "controlled_cer": mean(float(row["controlled"]["cer"]["rate"]) for row in scored_rows),  # type: ignore[index]
        "baseline_rtf": mean(float(row["baseline_rtf"]) for row in scored_rows),
        "controlled_rtf": mean(float(row["controlled_rtf"]) for row in scored_rows),
        "dominant_error_classes": dict(error_classes.most_common()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "optional_dependencies": dependency_status(),
    }


def error_examples(scored_rows: list[dict[str, object]], limit: int = 5) -> list[dict[str, object]]:
    rows = sorted(
        scored_rows,
        key=lambda row: float(row["baseline"]["wer"]["rate"]),  # type: ignore[index]
        reverse=True,
    )
    examples: list[dict[str, object]] = []
    for row in rows[:limit]:
        examples.append(
            {
                "audio_id": row["audio_id"],
                "condition": row["condition"],
                "error_class": row["error_class"],
                "reference_text": row["reference_text"],
                "baseline_hypothesis": row["baseline_hypothesis"],
                "controlled_hypothesis": row["controlled_hypothesis"],
                "baseline_wer": row["baseline"]["wer"]["rate"],  # type: ignore[index]
                "controlled_wer": row["controlled"]["wer"]["rate"],  # type: ignore[index]
            }
        )
    return examples


def synthesize_wav(path: Path, duration_seconds: float, frequency_hz: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    amplitude = 0.15
    frame_count = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            t = index / sample_rate
            sample = int(32767 * amplitude * math.sin(2 * math.pi * frequency_hz * t))
            frames.extend(struct.pack("<h", sample))
        handle.writeframes(bytes(frames))


def write_sample_data(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "audio_id": "lec-clean-01",
            "file": "lec-clean-01.wav",
            "duration": 0.80,
            "frequency": 440.0,
            "split": "validation",
            "language": "English",
            "condition": "clean lecture",
            "reference_text": "gradient descent updates the weights after each batch",
            "baseline_hypothesis": "gradient decent updates weights after each badge",
            "controlled_hypothesis": "gradient descent updates the weights after each batch",
            "baseline_processing_seconds": 0.18,
            "controlled_processing_seconds": 0.22,
            "error_class": "domain_vocabulary",
        },
        {
            "audio_id": "lec-noisy-02",
            "file": "lec-noisy-02.wav",
            "duration": 0.95,
            "frequency": 554.37,
            "split": "validation",
            "language": "English",
            "condition": "room echo",
            "reference_text": "the validation set guides model selection",
            "baseline_hypothesis": "validation set guide model selection",
            "controlled_hypothesis": "the validation set guides model selection",
            "baseline_processing_seconds": 0.20,
            "controlled_processing_seconds": 0.23,
            "error_class": "acoustic",
        },
        {
            "audio_id": "lec-term-03",
            "file": "lec-term-03.wav",
            "duration": 0.72,
            "frequency": 659.25,
            "split": "final_test",
            "language": "English",
            "condition": "domain term",
            "reference_text": "word error rate counts substitutions deletions and insertions",
            "baseline_hypothesis": "word error rate count substitutions deletion and insertion",
            "controlled_hypothesis": "word error rate counts substitutions deletions and insertions",
            "baseline_processing_seconds": 0.16,
            "controlled_processing_seconds": 0.21,
            "error_class": "domain_vocabulary",
        },
    ]
    manifest_path = output_dir / "sample_manifest.jsonl"
    rows = []
    for record in records:
        wav_path = output_dir / str(record["file"])
        synthesize_wav(wav_path, float(record["duration"]), float(record["frequency"]))
        row = {key: value for key, value in record.items() if key not in {"file", "duration", "frequency"}}
        row["path"] = str(wav_path.name)
        rows.append(row)
    write_jsonl(manifest_path, rows)
    return manifest_path


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    records = read_manifest(Path(args.manifest))
    scored_rows = [score_record(record) for record in records]
    summary = summarize(scored_rows)
    summary["normalization_rule"] = "lowercase; keep ASCII letters, digits, and apostrophes; collapse whitespace"
    summary["wall_clock_seconds"] = time.perf_counter() - started
    if args.save_artifacts:
        artifact_dir = Path(args.artifact_dir)
        write_jsonl(artifact_dir / "scored_segments.jsonl", scored_rows)
        write_json(artifact_dir / "run_summary.json", summary)
        write_jsonl(artifact_dir / "error_examples.jsonl", error_examples(scored_rows, args.error_limit))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-deps", action="store_true", help="Report optional dependency availability.")
    parser.add_argument("--allow-missing-deps", action="store_true", help="Do not fail when optional packages are absent.")
    parser.add_argument("--write-sample-data", type=Path, help="Write a tiny synthetic WAV sample dataset to this directory.")
    parser.add_argument("--run", action="store_true", help="Score an ASR manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("sample_audio/sample_manifest.jsonl"))
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/asr_eval"))
    parser.add_argument("--error-limit", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.check_deps:
        status = dependency_status()
        print(json.dumps(status, indent=2, sort_keys=True))
        if args.allow_missing_deps or all(status.values()):
            return 0
        return 1
    if args.write_sample_data:
        manifest_path = write_sample_data(args.write_sample_data)
        print(f"Wrote sample manifest: {manifest_path}")
        return 0
    if args.run:
        return run(args)
    print("No action requested. Use --check-deps, --write-sample-data, or --run.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
