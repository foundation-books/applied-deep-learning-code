#!/usr/bin/env python3
"""Benchmark sequence-length scaling for GRU, 1D CNN, and self-attention blocks."""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="none")
    parser.add_argument("--artifact-dir", default="runs/attention-scaling")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--allow-missing-deps", action="store_true")
    return parser.parse_args()


def load_torch(args: argparse.Namespace) -> Any | None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local environment
        message = f"Missing or unusable dependency: {exc}"
        if args.allow_missing_deps:
            print(message)
            print("Dependency check skipped because --allow-missing-deps was supplied.")
            return None
        raise RuntimeError(message) from exc
    return torch


def mps_available(torch: Any) -> bool:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(mps is not None and mps.is_available())


def select_device(args: argparse.Namespace, torch: Any) -> Any:
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    if args.device == "mps":
        if not mps_available(torch):
            raise RuntimeError(
                "--device mps requested, but PyTorch MPS is not available"
            )
        return torch.device("mps")
    if args.device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available(torch):
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        sync = getattr(mps, "synchronize", None)
        if callable(sync):
            sync()


def reset_peak_memory(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory(torch: Any, device: Any) -> dict[str, Any]:
    if device.type != "cuda":
        memory = {"cuda_available": False}
        if device.type == "mps":
            mps = getattr(torch, "mps", None)
            current = getattr(mps, "current_allocated_memory", None)
            driver = getattr(mps, "driver_allocated_memory", None)
            if callable(current):
                memory["mps_current_allocated_bytes"] = int(current())
            if callable(driver):
                memory["mps_driver_allocated_bytes"] = int(driver())
        return memory
    return {
        "cuda_available": True,
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def autocast_context(args: argparse.Namespace, torch: Any, device: Any) -> Any:
    if device.type != "cuda" or args.amp == "none":
        return nullcontext()
    dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def build_model(kind: str, args: argparse.Namespace, torch: Any) -> Any:
    if kind == "self_attention":
        return torch.nn.MultiheadAttention(args.width, args.heads, batch_first=True)
    if kind == "gru":
        return torch.nn.GRU(args.width, args.width, batch_first=True)
    if kind == "conv1d":
        return torch.nn.Sequential(
            torch.nn.Conv1d(args.width, args.width, kernel_size=5, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv1d(args.width, args.width, kernel_size=5, padding=2),
        )
    raise ValueError(f"unknown model kind: {kind}")


def run_model(kind: str, model: Any, inputs: Any, args: argparse.Namespace, torch: Any, device: Any) -> Any:
    with autocast_context(args, torch, device):
        if kind == "self_attention":
            output, _ = model(inputs, inputs, inputs, need_weights=False)
        elif kind == "gru":
            output, _ = model(inputs)
        elif kind == "conv1d":
            output = model(inputs.transpose(1, 2)).transpose(1, 2)
        else:
            raise ValueError(f"unknown model kind: {kind}")
        loss = output.float().square().mean()
    loss.backward()
    return loss


def benchmark_one(kind: str, length: int, args: argparse.Namespace, torch: Any, device: Any) -> dict[str, Any]:
    model = build_model(kind, args, torch).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    for _ in range(args.warmup):
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(args.batch_size, length, args.width, device=device)
        run_model(kind, model, inputs, args, torch, device)
        synchronize(torch, device)

    reset_peak_memory(torch, device)
    times = []
    for _ in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(args.batch_size, length, args.width, device=device)
        synchronize(torch, device)
        started_at = time.perf_counter()
        run_model(kind, model, inputs, args, torch, device)
        synchronize(torch, device)
        times.append(time.perf_counter() - started_at)

    memory = peak_memory(torch, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        empty_cache = getattr(mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    return {
        "kind": kind,
        "length": int(length),
        "batch_size": int(args.batch_size),
        "width": int(args.width),
        "heads": int(args.heads) if kind == "self_attention" else "",
        "seconds_mean": float(statistics.mean(times)),
        "seconds_median": float(statistics.median(times)),
        "seconds_min": float(min(times)),
        "seconds_max": float(max(times)),
        "tokens_per_second_mean": float(args.batch_size * length / statistics.mean(times)),
        "peak_memory": memory,
    }


def package_versions(torch: Any, device: Any) -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(getattr(torch, "__version__", "unknown")),
        "device": str(device),
    }
    if device.type == "cuda":
        versions["cuda_available"] = bool(torch.cuda.is_available())
        versions["cuda_device_name"] = torch.cuda.get_device_name(0)
        versions["cuda_version"] = str(getattr(torch.version, "cuda", "unknown"))
    else:
        versions["cuda_available"] = bool(torch.cuda.is_available())
    versions["mps_available"] = mps_available(torch)
    return versions


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "kind",
        "length",
        "batch_size",
        "width",
        "heads",
        "seconds_mean",
        "seconds_median",
        "tokens_per_second_mean",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            memory = row.get("peak_memory", {})
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fields},
                    "peak_allocated_bytes": memory.get("max_allocated_bytes", ""),
                    "peak_reserved_bytes": memory.get("max_reserved_bytes", ""),
                }
            )


def main() -> int:
    args = parse_args()
    torch = load_torch(args)
    if torch is None:
        return 0
    if args.check_deps:
        print("Dependency check passed.")
        return 0
    device = select_device(args, torch)
    rows = []
    for length in args.lengths:
        for kind in ("gru", "conv1d", "self_attention"):
            rows.append(benchmark_one(kind, length, args, torch, device))
    summary = {
        "command": " ".join(sys.argv),
        "versions": package_versions(torch, device),
        "amp": args.amp,
        "iterations": int(args.iterations),
        "warmup": int(args.warmup),
        "results": rows,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.save_artifacts:
        artifact_root = Path(args.artifact_dir)
        save_json(artifact_root / "attention_scaling_summary.json", summary)
        save_csv(artifact_root / "attention_scaling_table.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
