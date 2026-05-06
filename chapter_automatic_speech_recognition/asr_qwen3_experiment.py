#!/usr/bin/env python3
"""Qwen3-ASR experiment helper for the ASR chapter.

This script intentionally keeps heavyweight imports inside command handlers so
the repository smoke check can compile it without installing ASR runtimes. It
prepares a small public LibriSpeech-derived manifest, runs Qwen3-ASR offline or
streaming, builds the manifest expected by ``asr_transcription_eval.py``, and
can run Qwen3-ForcedAligner on selected hypotheses.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import datetime as dt
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable
from urllib.request import Request, urlopen
import uuid
import wave


DEFAULT_CONTEXT = (
    "Transcribe the speech exactly. Keep spoken words as text; do not translate. "
    "Course terms may include WER, CER, real-time factor, Qwen3-ASR, "
    "gradient descent, validation set, and transformer."
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return value


def public_metadata_value(value: Any) -> Any:
    safe = json_safe(value)
    home = str(Path.home())

    def sanitize(item: Any) -> Any:
        if isinstance(item, str):
            return item.replace(home, "<home>") if home else item
        if isinstance(item, dict):
            return {key: sanitize(inner) for key, inner in item.items()}
        if isinstance(item, list):
            return [sanitize(inner) for inner in item]
        return item

    return sanitize(safe)


def redact_cli_option(argv: Any, option: str, replacement: str = "<redacted>") -> Any:
    if not isinstance(argv, list):
        return argv
    redacted: list[Any] = []
    redact_next = False
    for item in argv:
        if redact_next:
            redacted.append(replacement)
            redact_next = False
            continue
        if item == option:
            redacted.append(item)
            redact_next = True
            continue
        if isinstance(item, str) and item.startswith(f"{option}="):
            redacted.append(f"{option}={replacement}")
            continue
        redacted.append(item)
    return redacted


def public_base_url(args: argparse.Namespace) -> str:
    if getattr(args, "record_base_url", False):
        return str(args.base_url)
    return "<redacted>"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            loaded = json.loads(stripped)
            if not isinstance(loaded, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(loaded)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def resolve_path(manifest_path: Path, value: object) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Manifest row has an empty path")
    path = Path(raw)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def audio_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def safe_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def command_record(args: argparse.Namespace) -> dict[str, Any]:
    public_args = public_metadata_value(vars(args))
    if isinstance(public_args, dict) and "base_url" in public_args:
        public_args["base_url"] = public_base_url(args)
    return {
        "argv": redact_cli_option(public_metadata_value(sys.argv), "--base-url"),
        "args": public_args,
        "created_utc": utc_now(),
        "hostname": "not-recorded",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def run_command(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def torch_dtype(torch: Any, name: str) -> Any:
    normalized = (name or "bfloat16").lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype for transformers backend: {name}")


def serialize_dataclass_like(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return serialize_dataclass_like(asdict(value))
    if isinstance(value, dict):
        return {str(k): serialize_dataclass_like(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_dataclass_like(v) for v in value]
    if hasattr(value, "items") and not isinstance(value, (str, bytes)):
        try:
            return [serialize_dataclass_like(v) for v in value.items]
        except Exception:
            pass
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return {k: serialize_dataclass_like(v) for k, v in vars(value).items()}
    return value


def condition_for_index(index: int) -> str:
    return "synthetic room noise" if index % 2 else "clean public speech"


def prepare_librispeech(args: argparse.Namespace) -> int:
    import torch
    import torchaudio

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    data_root.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    dataset = torchaudio.datasets.LIBRISPEECH(
        root=str(data_root),
        url=args.url,
        folder_in_archive="LibriSpeech",
        download=True,
    )

    records: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(int(args.seed))
    for index, item in enumerate(dataset):
        if len(records) >= args.limit:
            break

        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = item
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
            sample_rate = 16_000

        condition = condition_for_index(index)
        variant = "noisy" if condition.startswith("synthetic") else "clean"
        if variant == "noisy":
            noise = torch.randn(waveform.shape, generator=generator, dtype=waveform.dtype)
            signal_rms = torch.sqrt(torch.mean(waveform**2)).clamp_min(1e-6)
            noise_rms = torch.sqrt(torch.mean(noise**2)).clamp_min(1e-6)
            target_noise_rms = signal_rms / (10 ** (float(args.noise_snr_db) / 20.0))
            waveform = torch.clamp(waveform + noise * (target_noise_rms / noise_rms), -1.0, 1.0)

        split = "validation" if len(records) < args.validation_count else "final_test"
        audio_id = f"ls-{speaker_id}-{chapter_id}-{utterance_id}-{variant}"
        wav_path = audio_dir / f"{audio_id}.wav"
        torchaudio.save(str(wav_path), waveform, sample_rate)
        duration = audio_duration_seconds(wav_path)

        records.append(
            {
                "audio_id": audio_id,
                "path": str(wav_path.relative_to(output_dir)),
                "split": split,
                "language": "English",
                "condition": condition,
                "duration_seconds": duration,
                "reference_text": str(transcript).strip(),
                "baseline_hypothesis": "",
                "controlled_hypothesis": "",
                "baseline_processing_seconds": 0.0,
                "controlled_processing_seconds": 0.0,
                "error_class": "acoustic" if variant == "noisy" else "other",
                "source_dataset": f"LibriSpeech/{args.url}",
                "source_speaker_id": str(speaker_id),
                "source_chapter_id": str(chapter_id),
                "source_utterance_id": str(utterance_id),
            }
        )

    if len(records) < args.limit:
        raise RuntimeError(f"Only prepared {len(records)} record(s), expected {args.limit}")

    manifest_path = output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, records)
    write_json(
        output_dir / "provenance.json",
        {
            **command_record(args),
            "manifest": str(manifest_path),
            "record_count": len(records),
            "validation_count": sum(1 for row in records if row["split"] == "validation"),
            "final_test_count": sum(1 for row in records if row["split"] == "final_test"),
            "noise_snr_db": args.noise_snr_db,
        },
    )
    print(manifest_path)
    return 0


def load_qwen_model(args: argparse.Namespace) -> Any:
    import torch
    from qwen_asr import Qwen3ASRModel

    if args.backend == "transformers":
        kwargs: dict[str, Any] = {
            "dtype": torch_dtype(torch, args.dtype),
            "device_map": args.device_map,
        }
        if args.attn_implementation:
            kwargs["attn_implementation"] = args.attn_implementation
        return Qwen3ASRModel.from_pretrained(
            args.model_name,
            max_inference_batch_size=args.max_inference_batch_size,
            max_new_tokens=args.max_new_tokens,
            **kwargs,
        )

    kwargs = {
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    return Qwen3ASRModel.LLM(
        model=args.model_name,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
        **kwargs,
    )


def language_for_row(row: dict[str, Any], args: argparse.Namespace) -> str | None:
    if args.force_language:
        return args.force_language
    if args.language_from_manifest:
        language = str(row.get("language", "")).strip()
        return language or None
    return None


def run_qwen(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    rows = read_jsonl(manifest_path)
    if args.limit:
        rows = rows[: args.limit]

    model = load_qwen_model(args)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        audio_path = resolve_path(manifest_path, row.get("path"))
        duration = audio_duration_seconds(audio_path)
        language = language_for_row(row, args)
        started = time.perf_counter()
        error = ""
        text = ""
        detected_language = ""
        try:
            result = model.transcribe(
                audio=str(audio_path),
                context=args.context or "",
                language=language,
                return_time_stamps=False,
            )[0]
            text = str(getattr(result, "text", "")).strip()
            detected_language = str(getattr(result, "language", "")).strip()
        except Exception as exc:
            error = repr(exc)
            if not args.continue_on_error:
                raise
        processing_seconds = time.perf_counter() - started
        output_rows.append(
            {
                "audio_id": row.get("audio_id"),
                "path": str(row.get("path", "")),
                "split": row.get("split", ""),
                "condition": row.get("condition", ""),
                "reference_text": row.get("reference_text", ""),
                "run_id": args.run_id,
                "model_name": args.model_name,
                "backend": args.backend,
                "dtype": args.dtype,
                "context": args.context or "",
                "forced_language": language or "",
                "detected_language": detected_language,
                "hypothesis_text": text,
                "duration_seconds": duration,
                "processing_seconds": processing_seconds,
                "rtf": processing_seconds / duration if duration > 0 else math.nan,
                "error": error,
                "created_utc": utc_now(),
            }
        )
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "audio_id": row.get("audio_id"),
                    "seconds": round(processing_seconds, 3),
                    "rtf": round(processing_seconds / duration, 3) if duration > 0 else None,
                    "error": error,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    write_jsonl(Path(args.output_jsonl), output_rows)
    write_json(
        Path(args.metadata_json),
        {
            **command_record(args),
            "nvidia_smi": run_command(["nvidia-smi"]),
            "output_jsonl": str(args.output_jsonl),
        },
    )
    return 0


def load_audio_16k(path: Path) -> Any:
    import librosa
    import numpy as np
    import soundfile as sf

    wav, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    if sample_rate != 16_000:
        wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=16_000)
    return np.asarray(wav, dtype="float32")


def run_qwen_streaming(args: argparse.Namespace) -> int:
    if args.backend != "vllm":
        raise ValueError("Streaming Qwen3-ASR requires --backend vllm")

    manifest_path = Path(args.manifest)
    rows = read_jsonl(manifest_path)
    if args.limit:
        rows = rows[: args.limit]

    model = load_qwen_model(args)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        audio_path = resolve_path(manifest_path, row.get("path"))
        wav = load_audio_16k(audio_path)
        duration = audio_duration_seconds(audio_path)
        language = language_for_row(row, args)
        chunk_samples = max(1, int(round(args.chunk_size_sec * 16_000)))
        state = model.init_streaming_state(
            context=args.context or "",
            language=language,
            unfixed_chunk_num=args.unfixed_chunk_num,
            unfixed_token_num=args.unfixed_token_num,
            chunk_size_sec=args.chunk_size_sec,
        )

        started = time.perf_counter()
        first_partial_seconds = math.nan
        error = ""
        try:
            for offset in range(0, len(wav), chunk_samples):
                state = model.streaming_transcribe(wav[offset : offset + chunk_samples], state)
                if state.text.strip() and math.isnan(first_partial_seconds):
                    first_partial_seconds = time.perf_counter() - started
            finalize_started = time.perf_counter()
            state = model.finish_streaming_transcribe(state)
            finalization_seconds = time.perf_counter() - finalize_started
        except Exception as exc:
            error = repr(exc)
            finalization_seconds = math.nan
            if not args.continue_on_error:
                raise

        processing_seconds = time.perf_counter() - started
        output_rows.append(
            {
                "audio_id": row.get("audio_id"),
                "path": str(row.get("path", "")),
                "split": row.get("split", ""),
                "condition": row.get("condition", ""),
                "reference_text": row.get("reference_text", ""),
                "run_id": args.run_id,
                "model_name": args.model_name,
                "backend": "vllm-streaming",
                "dtype": args.dtype,
                "context": args.context or "",
                "forced_language": language or "",
                "detected_language": getattr(state, "language", ""),
                "hypothesis_text": getattr(state, "text", "").strip(),
                "duration_seconds": duration,
                "processing_seconds": processing_seconds,
                "rtf": processing_seconds / duration if duration > 0 else math.nan,
                "first_partial_seconds": first_partial_seconds,
                "finalization_seconds": finalization_seconds,
                "chunk_size_sec": args.chunk_size_sec,
                "chunks_processed": getattr(state, "chunk_id", 0),
                "error": error,
                "created_utc": utc_now(),
            }
        )
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "audio_id": row.get("audio_id"),
                    "seconds": round(processing_seconds, 3),
                    "first_partial": (
                        None
                        if math.isnan(first_partial_seconds)
                        else round(first_partial_seconds, 3)
                    ),
                    "error": error,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    write_jsonl(Path(args.output_jsonl), output_rows)
    write_json(
        Path(args.metadata_json),
        {
            **command_record(args),
            "nvidia_smi": run_command(["nvidia-smi"]),
            "output_jsonl": str(args.output_jsonl),
        },
    )
    return 0


def multipart_body(
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
) -> tuple[bytes, str]:
    boundary = f"----adl-asr-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    chunks.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_path.name}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(file_path.read_bytes())
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), boundary


def post_transcription_api(
    base_url: str,
    audio_path: Path,
    model_name: str,
    language: str | None,
    context: str,
    stream: bool,
    timeout: float,
) -> tuple[str, str, float, float]:
    fields = {
        "model": model_name,
        "prompt": context or "",
        "stream": "true" if stream else "false",
    }
    if language:
        fields["language"] = language
    body, boundary = multipart_body(fields, "file", audio_path)
    url = base_url.rstrip("/") + "/v1/audio/transcriptions"
    request = Request(
        url=url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    started = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        if not stream:
            payload = json.loads(response.read().decode("utf-8"))
            return (
                str(payload.get("text", "")).strip(),
                str(payload.get("language", "")).strip(),
                time.perf_counter() - started,
                math.nan,
            )

        parts: list[str] = []
        first_partial_seconds = math.nan
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            payload = json.loads(data)
            choices = payload.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = str(delta.get("content") or "")
            if content:
                if math.isnan(first_partial_seconds):
                    first_partial_seconds = time.perf_counter() - started
                parts.append(content)
        return ("".join(parts).strip(), "", time.perf_counter() - started, first_partial_seconds)


def run_openai_api(args: argparse.Namespace) -> int:
    normalize_context_args(args)
    manifest_path = Path(args.manifest)
    rows = read_jsonl(manifest_path)
    if args.limit:
        rows = rows[: args.limit]

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        audio_path = resolve_path(manifest_path, row.get("path"))
        duration = audio_duration_seconds(audio_path)
        language = language_for_row(row, args)
        error = ""
        text = ""
        detected_language = ""
        processing_seconds = math.nan
        first_partial_seconds = math.nan
        try:
            text, detected_language, processing_seconds, first_partial_seconds = post_transcription_api(
                base_url=args.base_url,
                audio_path=audio_path,
                model_name=args.model_name,
                language=language,
                context=args.context or "",
                stream=args.stream,
                timeout=args.timeout,
            )
        except Exception as exc:
            error = repr(exc)
            if not args.continue_on_error:
                raise

        output_rows.append(
            {
                "audio_id": row.get("audio_id"),
                "path": str(row.get("path", "")),
                "split": row.get("split", ""),
                "condition": row.get("condition", ""),
                "reference_text": row.get("reference_text", ""),
                "run_id": args.run_id,
                "model_name": args.model_name,
                "backend": "openai-api-streaming" if args.stream else "openai-api",
                "base_url": public_base_url(args),
                "context": args.context or "",
                "forced_language": language or "",
                "detected_language": detected_language,
                "hypothesis_text": text,
                "duration_seconds": duration,
                "processing_seconds": processing_seconds,
                "rtf": processing_seconds / duration if duration > 0 else math.nan,
                "first_partial_seconds": first_partial_seconds,
                "error": error,
                "created_utc": utc_now(),
            }
        )
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "audio_id": row.get("audio_id"),
                    "seconds": None if math.isnan(processing_seconds) else round(processing_seconds, 3),
                    "first_partial": (
                        None if math.isnan(first_partial_seconds) else round(first_partial_seconds, 3)
                    ),
                    "error": error,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    write_jsonl(Path(args.output_jsonl), output_rows)
    write_json(
        Path(args.metadata_json),
        {
            **command_record(args),
            "output_jsonl": str(args.output_jsonl),
        },
    )
    return 0


def rows_by_run(path: Path, run_id: str) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if str(row.get("run_id")) != run_id:
            continue
        audio_id = str(row.get("audio_id", "")).strip()
        if audio_id:
            selected[audio_id] = row
    if not selected:
        raise ValueError(f"No rows found for run_id={run_id!r} in {path}")
    return selected


def build_scoring_manifest(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    baseline = rows_by_run(Path(args.hypotheses_jsonl), args.baseline_run_id)
    controlled = rows_by_run(Path(args.hypotheses_jsonl), args.controlled_run_id)
    rows = []
    for row in read_jsonl(manifest_path):
        audio_id = str(row.get("audio_id", "")).strip()
        if audio_id not in baseline or audio_id not in controlled:
            if args.require_all:
                raise ValueError(f"Missing hypothesis for audio_id={audio_id}")
            continue
        updated = dict(row)
        base_row = baseline[audio_id]
        controlled_row = controlled[audio_id]
        updated["path"] = str(row.get("path", ""))
        updated["baseline_hypothesis"] = str(base_row.get("hypothesis_text", ""))
        updated["controlled_hypothesis"] = str(controlled_row.get("hypothesis_text", ""))
        updated["baseline_processing_seconds"] = safe_float(
            base_row.get("processing_seconds"), default=0.0
        )
        updated["controlled_processing_seconds"] = safe_float(
            controlled_row.get("processing_seconds"), default=0.0
        )
        rows.append(updated)

    write_jsonl(Path(args.output_manifest), rows)
    write_json(
        Path(args.metadata_json),
        {
            **command_record(args),
            "baseline_run_id": args.baseline_run_id,
            "controlled_run_id": args.controlled_run_id,
            "row_count": len(rows),
            "output_manifest": str(args.output_manifest),
        },
    )
    print(args.output_manifest)
    return 0


def run_aligner(args: argparse.Namespace) -> int:
    import torch
    from qwen_asr import Qwen3ForcedAligner

    manifest_path = Path(args.manifest)
    manifest_rows = {str(row.get("audio_id")): row for row in read_jsonl(manifest_path)}
    hypotheses = rows_by_run(Path(args.hypotheses_jsonl), args.source_run_id)
    aligner = Qwen3ForcedAligner.from_pretrained(
        args.aligner_model,
        dtype=torch_dtype(torch, args.dtype),
        device_map=args.device_map,
    )

    output_rows: list[dict[str, Any]] = []
    for audio_id, hyp in hypotheses.items():
        manifest_row = manifest_rows.get(audio_id)
        if manifest_row is None:
            continue
        audio_path = resolve_path(manifest_path, manifest_row.get("path"))
        text = (
            str(hyp.get("hypothesis_text", "")).strip()
            if args.align_hypothesis
            else str(manifest_row.get("reference_text", "")).strip()
        )
        language = str(hyp.get("detected_language") or manifest_row.get("language") or "English")
        started = time.perf_counter()
        error = ""
        items: Any = []
        try:
            aligned = aligner.align(audio=str(audio_path), text=text, language=language)[0]
            items = serialize_dataclass_like(aligned)
        except Exception as exc:
            error = repr(exc)
            if not args.continue_on_error:
                raise
        output_rows.append(
            {
                "audio_id": audio_id,
                "path": str(manifest_row.get("path", "")),
                "source_run_id": args.source_run_id,
                "aligner_model": args.aligner_model,
                "aligned_text_source": "hypothesis" if args.align_hypothesis else "reference",
                "language": language,
                "text": text,
                "alignment": items,
                "processing_seconds": time.perf_counter() - started,
                "error": error,
                "created_utc": utc_now(),
            }
        )
        print(json.dumps({"audio_id": audio_id, "alignment_error": error}, sort_keys=True), flush=True)

    write_jsonl(Path(args.output_jsonl), output_rows)
    write_json(Path(args.metadata_json), {**command_record(args), "row_count": len(output_rows)})
    return 0


def add_prepare_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("prepare-librispeech")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="test-clean")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--noise-snr-db", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.set_defaults(func=prepare_librispeech)


def add_qwen_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--max-inference-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--context", default="")
    parser.add_argument("--default-domain-context", action="store_true")
    parser.add_argument("--force-language", default="")
    parser.add_argument("--language-from-manifest", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")


def normalize_context_args(args: argparse.Namespace) -> None:
    if getattr(args, "default_domain_context", False) and not args.context:
        args.context = DEFAULT_CONTEXT


def run_qwen_with_context(args: argparse.Namespace) -> int:
    normalize_context_args(args)
    return run_qwen(args)


def run_qwen_streaming_with_context(args: argparse.Namespace) -> int:
    normalize_context_args(args)
    return run_qwen_streaming(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_prepare_parser(subparsers)

    run_parser = subparsers.add_parser("run-qwen")
    add_qwen_common(run_parser)
    run_parser.set_defaults(func=run_qwen_with_context)

    streaming_parser = subparsers.add_parser("run-streaming")
    add_qwen_common(streaming_parser)
    streaming_parser.add_argument("--chunk-size-sec", type=float, default=2.0)
    streaming_parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    streaming_parser.add_argument("--unfixed-token-num", type=int, default=5)
    streaming_parser.set_defaults(func=run_qwen_streaming_with_context)

    api_parser = subparsers.add_parser("run-openai-api")
    add_qwen_common(api_parser)
    api_parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    api_parser.add_argument(
        "--record-base-url",
        action="store_true",
        help="Record the API endpoint in output artifacts. By default it is redacted.",
    )
    api_parser.add_argument("--timeout", type=float, default=180.0)
    api_parser.add_argument("--stream", action="store_true")
    api_parser.set_defaults(func=run_openai_api)

    score_parser = subparsers.add_parser("build-scoring-manifest")
    score_parser.add_argument("--manifest", type=Path, required=True)
    score_parser.add_argument("--hypotheses-jsonl", type=Path, required=True)
    score_parser.add_argument("--baseline-run-id", required=True)
    score_parser.add_argument("--controlled-run-id", required=True)
    score_parser.add_argument("--output-manifest", type=Path, required=True)
    score_parser.add_argument("--metadata-json", type=Path, required=True)
    score_parser.add_argument("--require-all", action="store_true")
    score_parser.set_defaults(func=build_scoring_manifest)

    align_parser = subparsers.add_parser("run-aligner")
    align_parser.add_argument("--manifest", type=Path, required=True)
    align_parser.add_argument("--hypotheses-jsonl", type=Path, required=True)
    align_parser.add_argument("--source-run-id", required=True)
    align_parser.add_argument("--output-jsonl", type=Path, required=True)
    align_parser.add_argument("--metadata-json", type=Path, required=True)
    align_parser.add_argument("--aligner-model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    align_parser.add_argument("--dtype", default="bfloat16")
    align_parser.add_argument("--device-map", default="cuda:0")
    align_parser.add_argument("--align-hypothesis", action="store_true")
    align_parser.add_argument("--continue-on-error", action="store_true")
    align_parser.set_defaults(func=run_aligner)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
