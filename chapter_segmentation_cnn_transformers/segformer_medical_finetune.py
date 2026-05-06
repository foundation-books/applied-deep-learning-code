"""Fine-tune SegFormer for a small binary medical segmentation dataset.

The script expects a CSV file with columns ``image_path``, ``mask_path``, and
``split``. Paths may be absolute or relative to ``--data-root``. The intended
chapter workflow uses Kvasir-SEG, but any two-dimensional binary medical
segmentation dataset with image-mask pairs can use the same interface.

Heavy machine-learning libraries are imported only inside the training command
so repository smoke checks can run without installing optional dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Iterable


VALID_SPLITS = {"train", "validation", "val", "test"}
REQUIRED_TRAINING_PACKAGES = ("torch", "transformers", "PIL", "numpy")
OPTIONAL_PACKAGES = ("matplotlib", "plotnine")
DISTRIBUTION_NAMES = {"PIL": "Pillow"}


def load_plot_style() -> Any:
    for parent in Path(__file__).resolve().parents:
        for candidate in (
            parent / "adl_plot_style.py",
            parent / "book" / "_maintenance" / "plot_style.py",
        ):
            if not candidate.exists():
                continue
            spec = importlib_util.spec_from_file_location("adl_plot_style", candidate)
            if spec is None or spec.loader is None:
                break
            module = importlib_util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Could not locate adl_plot_style.py")


PLOT_STYLE = load_plot_style()


@dataclass(frozen=True)
class SegmentationRecord:
    image_path: Path
    mask_path: Path
    split: str


@dataclass(frozen=True)
class MetricCounts:
    intersection: int
    union: int
    true_pixels: int
    predicted_pixels: int
    total_pixels: int
    correct_pixels: int

    @property
    def dice(self) -> float:
        denominator = self.true_pixels + self.predicted_pixels
        return 1.0 if denominator == 0 else (2.0 * self.intersection) / denominator

    @property
    def iou(self) -> float:
        return 1.0 if self.union == 0 else self.intersection / self.union

    @property
    def pixel_accuracy(self) -> float:
        return (
            0.0 if self.total_pixels == 0 else self.correct_pixels / self.total_pixels
        )


def package_available(name: str) -> bool:
    return importlib_util.find_spec(name) is not None


def package_version(name: str) -> str | None:
    distribution = DISTRIBUTION_NAMES.get(name, name)
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def dependency_report() -> dict[str, bool]:
    names = tuple(REQUIRED_TRAINING_PACKAGES) + tuple(OPTIONAL_PACKAGES)
    return {name: package_available(name) for name in names}


def package_versions() -> dict[str, str | None]:
    names = tuple(REQUIRED_TRAINING_PACKAGES) + tuple(OPTIONAL_PACKAGES)
    return {name: package_version(name) for name in names}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_key_value_csv(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in sorted(payload):
            value = payload[key]
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            writer.writerow({"metric": key, "value": value})


def command_dependency_check(args: argparse.Namespace) -> int:
    availability = dependency_report()
    missing = [name for name in REQUIRED_TRAINING_PACKAGES if not availability[name]]
    payload = {
        "required": list(REQUIRED_TRAINING_PACKAGES),
        "optional": list(OPTIONAL_PACKAGES),
        "available": availability,
        "missing_required": missing,
    }
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "segformer_dependency_check.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2))
    if missing and not args.allow_missing_deps:
        raise SystemExit(f"Missing required training packages: {', '.join(missing)}")
    return 0


def resolve_path(path_text: str, base: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def normalize_split(split: str) -> str:
    value = split.strip().lower()
    if value == "val":
        return "validation"
    if value not in VALID_SPLITS:
        raise ValueError(
            f"Unsupported split {split!r}; expected train, validation, or test"
        )
    return value


def read_pairs_csv(pairs_csv: Path, data_root: Path | None) -> list[SegmentationRecord]:
    base = data_root.resolve() if data_root is not None else pairs_csv.resolve().parent
    rows = list(csv.DictReader(pairs_csv.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"{pairs_csv} has no rows")
    required = {"image_path", "mask_path", "split"}
    missing_columns = sorted(required - set(rows[0].keys()))
    if missing_columns:
        raise ValueError(
            f"{pairs_csv} is missing columns: {', '.join(missing_columns)}"
        )
    records = [
        SegmentationRecord(
            image_path=resolve_path(row["image_path"], base),
            mask_path=resolve_path(row["mask_path"], base),
            split=normalize_split(row["split"]),
        )
        for row in rows
    ]
    return records


def split_records(
    records: Iterable[SegmentationRecord],
) -> dict[str, list[SegmentationRecord]]:
    by_split = {"train": [], "validation": [], "test": []}
    for record in records:
        by_split[record.split].append(record)
    return by_split


def dataset_provenance_payload(
    records: list[SegmentationRecord],
    by_split: dict[str, list[SegmentationRecord]],
    pairs_csv: Path,
    data_root: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "dataset_name": args.dataset_name,
        "dataset_citation": args.dataset_citation,
        "dataset_url": args.dataset_url,
        "pairs_csv": str(pairs_csv),
        "data_root": str(data_root) if data_root else None,
        "split_rule": args.split_rule,
        "split_counts": {split: len(items) for split, items in by_split.items()},
        "record_count": len(records),
        "seed": args.seed,
        "sample_records": [
            {
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path),
                "split": record.split,
            }
            for record in records[:10]
        ],
    }


def environment_payload(
    args: argparse.Namespace,
    device: str | None = None,
    deps: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages_available": dependency_report(),
        "package_versions": package_versions(),
        "device": device,
        "device_requested": getattr(args, "device", None),
        "amp_requested": bool(args.amp),
        "cpu_forced": bool(args.cpu),
    }
    if deps is not None:
        torch = deps["torch"]
        cuda_available = bool(torch.cuda.is_available())
        mps_is_available = mps_available(torch)
        payload.update(
            {
                "torch_version": getattr(torch, "__version__", None),
                "cuda_available": cuda_available,
                "cuda_device_count": int(torch.cuda.device_count())
                if cuda_available
                else 0,
                "cuda_device_name": torch.cuda.get_device_name(0)
                if cuda_available
                else None,
                "mps_available": mps_is_available,
                "amp_enabled": bool(args.amp and device == "cuda"),
            }
        )
    return payload


def mps_available(torch: Any) -> bool:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(mps is not None and mps.is_available())


def resolve_device(torch: Any, requested: str, force_cpu: bool = False) -> str:
    if force_cpu or requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return "cuda"
    if requested == "mps":
        if not mps_available(torch):
            raise RuntimeError(
                "--device mps requested, but PyTorch MPS is not available"
            )
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if mps_available(torch):
        return "mps"
    return "cpu"


def autocast_context(torch: Any, device: str, amp_requested: bool) -> Any:
    if amp_requested and device == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def metric_counts_from_binary_masks(
    true_values: Iterable[int], pred_values: Iterable[int]
) -> MetricCounts:
    true = [1 if int(value) else 0 for value in true_values]
    pred = [1 if int(value) else 0 for value in pred_values]
    if len(true) != len(pred):
        raise ValueError("true and predicted masks must have the same number of pixels")
    intersection = sum(1 for t, p in zip(true, pred) if t == 1 and p == 1)
    true_pixels = sum(true)
    pred_pixels = sum(pred)
    union = true_pixels + pred_pixels - intersection
    correct = sum(1 for t, p in zip(true, pred) if t == p)
    return MetricCounts(
        intersection=intersection,
        union=union,
        true_pixels=true_pixels,
        predicted_pixels=pred_pixels,
        total_pixels=len(true),
        correct_pixels=correct,
    )


def import_training_dependencies() -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    return {
        "np": np,
        "torch": torch,
        "F": F,
        "Image": Image,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "AutoImageProcessor": AutoImageProcessor,
        "SegformerForSemanticSegmentation": SegformerForSemanticSegmentation,
    }


def set_reproducibility(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset_class(
    deps: dict[str, Any], image_size: int, mean: list[float], std: list[float]
) -> Any:
    np = deps["np"]
    torch = deps["torch"]
    Image = deps["Image"]
    Dataset = deps["Dataset"]

    class PairDataset(Dataset):  # type: ignore[misc, valid-type]
        def __init__(self, records: list[SegmentationRecord]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            record = self.records[index]
            image = Image.open(record.image_path).convert("RGB")
            mask = Image.open(record.mask_path).convert("L")
            image = image.resize(
                (image_size, image_size), resample=Image.Resampling.BILINEAR
            )
            mask = mask.resize(
                (image_size, image_size), resample=Image.Resampling.NEAREST
            )

            image_array = np.asarray(image, dtype=np.float32) / 255.0
            image_array = (
                image_array - np.asarray(mean, dtype=np.float32)
            ) / np.asarray(std, dtype=np.float32)
            image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()

            mask_array = np.asarray(mask, dtype=np.uint8)
            labels = torch.from_numpy((mask_array > 0).astype(np.int64)).long()
            return {
                "pixel_values": image_tensor,
                "labels": labels,
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path),
            }

    return PairDataset


def segmentation_loss(
    logits: Any,
    labels: Any,
    deps: dict[str, Any],
    dice_loss_weight: float,
    smooth: float = 1.0,
) -> tuple[Any, dict[str, float]]:
    torch = deps["torch"]
    F = deps["F"]
    resized_logits = F.interpolate(
        logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    ce_loss = F.cross_entropy(resized_logits, labels)
    probabilities = torch.softmax(resized_logits, dim=1)[:, 1]
    target = (labels == 1).float()
    intersection = (probabilities * target).sum(dim=(1, 2))
    denominator = probabilities.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    dice_loss = 1.0 - dice.mean()
    loss = ce_loss + dice_loss_weight * dice_loss
    return loss, {
        "cross_entropy_loss": float(ce_loss.detach().cpu().item()),
        "dice_loss": float(dice_loss.detach().cpu().item()),
    }


def counts_for_threshold(
    logits: Any, labels: Any, threshold: float, deps: dict[str, Any]
) -> MetricCounts:
    torch = deps["torch"]
    F = deps["F"]
    resized_logits = F.interpolate(
        logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    probabilities = torch.softmax(resized_logits, dim=1)[:, 1]
    pred = probabilities >= threshold
    true = labels == 1
    intersection = int((pred & true).sum().detach().cpu().item())
    true_pixels = int(true.sum().detach().cpu().item())
    pred_pixels = int(pred.sum().detach().cpu().item())
    union = true_pixels + pred_pixels - intersection
    correct = int((pred == true).sum().detach().cpu().item())
    total = int(true.numel())
    return MetricCounts(
        intersection=intersection,
        union=union,
        true_pixels=true_pixels,
        predicted_pixels=pred_pixels,
        total_pixels=total,
        correct_pixels=correct,
    )


def add_counts(left: MetricCounts, right: MetricCounts) -> MetricCounts:
    return MetricCounts(
        intersection=left.intersection + right.intersection,
        union=left.union + right.union,
        true_pixels=left.true_pixels + right.true_pixels,
        predicted_pixels=left.predicted_pixels + right.predicted_pixels,
        total_pixels=left.total_pixels + right.total_pixels,
        correct_pixels=left.correct_pixels + right.correct_pixels,
    )


def counts_to_metrics(counts: MetricCounts) -> dict[str, float | int]:
    return {
        **asdict(counts),
        "dice": counts.dice,
        "iou": counts.iou,
        "pixel_accuracy": counts.pixel_accuracy,
    }


def evaluate_model(
    model: Any,
    loader: Any,
    thresholds: list[float],
    deps: dict[str, Any],
    device: str,
    dice_loss_weight: float,
) -> dict[str, Any]:
    torch = deps["torch"]
    model.eval()
    totals = {
        threshold: MetricCounts(
            intersection=0,
            union=0,
            true_pixels=0,
            predicted_pixels=0,
            total_pixels=0,
            correct_pixels=0,
        )
        for threshold in thresholds
    }
    losses: list[float] = []
    image_count = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values)
            loss, _ = segmentation_loss(outputs.logits, labels, deps, dice_loss_weight)
            losses.append(float(loss.detach().cpu().item()))
            image_count += int(labels.shape[0])
            for threshold in thresholds:
                totals[threshold] = add_counts(
                    totals[threshold],
                    counts_for_threshold(outputs.logits, labels, threshold, deps),
                )
    elapsed = time.perf_counter() - start
    threshold_metrics = {
        f"{threshold:.3f}": counts_to_metrics(counts)
        for threshold, counts in totals.items()
    }
    best_threshold = max(thresholds, key=lambda value: totals[value].dice)
    return {
        "loss": sum(losses) / len(losses) if losses else math.nan,
        "threshold_metrics": threshold_metrics,
        "best_threshold": best_threshold,
        "best_metrics": counts_to_metrics(totals[best_threshold]),
        "seconds": elapsed,
        "images": image_count,
        "milliseconds_per_image": (1000.0 * elapsed / image_count)
        if image_count
        else math.nan,
    }


def split_metrics_payload(
    split: str,
    evaluation: dict[str, Any],
    best_epoch: int,
    selected_threshold: float,
) -> dict[str, Any]:
    payload = {
        "split": split,
        "loss": evaluation["loss"],
        "best_epoch": best_epoch,
        "selected_threshold": selected_threshold,
        "seconds": evaluation["seconds"],
        "images": evaluation["images"],
        "milliseconds_per_image": evaluation["milliseconds_per_image"],
    }
    payload.update(evaluation["best_metrics"])
    return payload


def write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_training_plot(history: list[dict[str, Any]], output_dir: Path) -> None:
    if not package_available("plotnine"):
        return
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd
    import plotnine as p9
    from plotnine import aes, facet_wrap, geom_line, geom_point, ggplot, labs

    if not history:
        return
    records = []
    for panel, values in [
        ("Training loss", [row["train_loss"] for row in history]),
        ("Validation Dice", [row["val_dice"] for row in history]),
    ]:
        for row, value in zip(history, values):
            records.append({"epoch": row["epoch"], "panel": panel, "value": value})
    data = pd.DataFrame.from_records(records)
    data["panel"] = pd.Categorical(
        data["panel"], categories=["Training loss", "Validation Dice"], ordered=True
    )
    plot = (
        ggplot(data, aes("epoch", "value"))
        + geom_line(size=0.8)
        + geom_point(size=2.1)
        + facet_wrap("~panel", scales="free_y", nrow=1)
        + labs(x="Epoch", y="metric value")
        + PLOT_STYLE.plot_theme(p9)
    )
    plot.save(
        output_dir / "training_curve.png",
        width=6,
        height=4,
        units="in",
        dpi=PLOT_STYLE.PLOT_DPI,
        verbose=False,
    )


def prepare_model(args: argparse.Namespace, deps: dict[str, Any]) -> tuple[Any, Any]:
    AutoImageProcessor = deps["AutoImageProcessor"]
    SegformerForSemanticSegmentation = deps["SegformerForSemanticSegmentation"]
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "background", 1: "target"},
        label2id={"background": 0, "target": 1},
        ignore_mismatched_sizes=True,
    )
    return model, processor


def write_prediction_grids(
    model: Any,
    records: list[SegmentationRecord],
    deps: dict[str, Any],
    image_size: int,
    mean: list[float],
    std: list[float],
    threshold: float,
    device: str,
    output_dir: Path,
    count: int,
) -> None:
    if not records:
        return
    np = deps["np"]
    torch = deps["torch"]
    F = deps["F"]
    Image = deps["Image"]

    panel_rows: list[tuple[float, Any]] = []
    model.eval()
    with torch.no_grad():
        for record in records[: max(count * 3, count)]:
            image = Image.open(record.image_path).convert("RGB")
            mask = Image.open(record.mask_path).convert("L")
            image = image.resize(
                (image_size, image_size), resample=Image.Resampling.BILINEAR
            )
            mask = mask.resize(
                (image_size, image_size), resample=Image.Resampling.NEAREST
            )

            image_array = np.asarray(image, dtype=np.float32) / 255.0
            normalized = (
                image_array - np.asarray(mean, dtype=np.float32)
            ) / np.asarray(std, dtype=np.float32)
            tensor = (
                torch.from_numpy(normalized.transpose(2, 0, 1))
                .float()
                .unsqueeze(0)
                .to(device)
            )
            logits = model(pixel_values=tensor).logits
            logits = F.interpolate(
                logits,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )
            probability = torch.softmax(logits, dim=1)[0, 1].detach().cpu().numpy()
            pred = probability >= threshold
            true = np.asarray(mask, dtype=np.uint8) > 0
            counts = metric_counts_from_binary_masks(true.reshape(-1), pred.reshape(-1))

            rgb = np.asarray(image, dtype=np.uint8)
            gt = np.zeros_like(rgb)
            gt[..., 0] = true.astype(np.uint8) * 255
            pred_rgb = np.zeros_like(rgb)
            pred_rgb[..., 1] = pred.astype(np.uint8) * 255
            prob_rgb = np.stack(
                [(probability * 255).astype(np.uint8)] * 3,
                axis=-1,
            )
            overlay = rgb.copy()
            overlay = np.where(
                true[..., None], (0.6 * overlay + 0.4 * gt).astype(np.uint8), overlay
            )
            overlay = np.where(
                pred[..., None],
                (0.6 * overlay + 0.4 * pred_rgb).astype(np.uint8),
                overlay,
            )
            row = np.concatenate([rgb, gt, prob_rgb, pred_rgb, overlay], axis=1)
            panel_rows.append((counts.dice, Image.fromarray(row)))

    if not panel_rows:
        return
    chosen = panel_rows[:count]
    failures = sorted(panel_rows, key=lambda item: item[0])[:count]
    for name, rows in (
        ("qualitative_grid.png", chosen),
        ("failure_grid.png", failures),
    ):
        width = rows[0][1].width
        height = rows[0][1].height * len(rows)
        grid = Image.new("RGB", (width, height), "white")
        for row_index, (_, panel) in enumerate(rows):
            grid.paste(panel, (0, row_index * panel.height))
        grid.save(output_dir / name)


def command_train(args: argparse.Namespace) -> int:
    pairs_csv = Path(args.pairs_csv).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    records = read_pairs_csv(pairs_csv, data_root)
    by_split = split_records(records)
    if not by_split["train"]:
        raise SystemExit("pairs CSV must contain at least one train row")
    if not by_split["validation"]:
        raise SystemExit("pairs CSV must contain at least one validation row")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: value for key, value in vars(args).items() if key != "func"
    }
    config = {
        **serializable_args,
        "pairs_csv": str(pairs_csv),
        "data_root": str(data_root) if data_root else None,
        "split_counts": {split: len(items) for split, items in by_split.items()},
    }
    write_json(output_dir / "config.json", config)
    write_json(
        output_dir / "dataset_provenance.json",
        dataset_provenance_payload(records, by_split, pairs_csv, data_root, args),
    )
    write_json(
        output_dir / "environment.json",
        environment_payload(args, device="dry-run" if args.dry_run else None),
    )

    if args.dry_run:
        print(
            f"Validated {len(records)} records and wrote run metadata to {output_dir}"
        )
        return 0

    deps = import_training_dependencies()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    set_reproducibility(args.seed, torch)
    device = resolve_device(torch, args.device, args.cpu)
    write_json(output_dir / "environment.json", environment_payload(args, device, deps))
    model, processor = prepare_model(args, deps)
    model.to(device)
    if args.amp and device != "cuda":
        print(
            f"[INFO] --amp is only enabled on CUDA; "
            f"running full precision on {device}."
        )

    mean = list(getattr(processor, "image_mean", [0.485, 0.456, 0.406]))
    std = list(getattr(processor, "image_std", [0.229, 0.224, 0.225]))
    PairDataset = build_dataset_class(deps, args.image_size, mean, std)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        PairDataset(by_split["train"]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        PairDataset(by_split["validation"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = (
        DataLoader(
            PairDataset(by_split["test"]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if by_split["test"]
        else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp_enabled = args.amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    thresholds = sorted(set(args.thresholds))
    history: list[dict[str, Any]] = []
    best_val_dice = -1.0
    best_epoch = 0
    best_threshold = thresholds[0]
    best_state: dict[str, Any] | None = None
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_ce: list[float] = []
        epoch_dice_loss: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            with autocast_context(torch, device, args.amp):
                outputs = model(pixel_values=pixel_values)
                loss, loss_parts = segmentation_loss(
                    outputs.logits,
                    labels,
                    deps,
                    dice_loss_weight=args.dice_loss_weight,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu().item()))
            epoch_ce.append(loss_parts["cross_entropy_loss"])
            epoch_dice_loss.append(loss_parts["dice_loss"])

        validation = evaluate_model(
            model,
            validation_loader,
            thresholds,
            deps,
            device,
            dice_loss_weight=args.dice_loss_weight,
        )
        val_metrics = validation["best_metrics"]
        row = {
            "epoch": epoch,
            "train_loss": sum(epoch_losses) / len(epoch_losses),
            "train_cross_entropy_loss": sum(epoch_ce) / len(epoch_ce),
            "train_dice_loss": sum(epoch_dice_loss) / len(epoch_dice_loss),
            "val_loss": validation["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "best_threshold": validation["best_threshold"],
            "val_ms_per_image": validation["milliseconds_per_image"],
        }
        history.append(row)
        print(
            "epoch {epoch}: train_loss={train_loss:.4f} "
            "val_dice={val_dice:.4f} val_iou={val_iou:.4f} threshold={best_threshold:.2f}".format(
                **row
            )
        )
        if float(val_metrics["dice"]) > best_val_dice:
            best_val_dice = float(val_metrics["dice"])
            best_epoch = epoch
            best_threshold = float(validation["best_threshold"])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if args.save_model:
                model.save_pretrained(output_dir / "best_model")
                processor.save_pretrained(output_dir / "best_model")

    train_seconds = time.perf_counter() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    final_validation = evaluate_model(
        model,
        validation_loader,
        [best_threshold],
        deps,
        device,
        dice_loss_weight=args.dice_loss_weight,
    )
    final_test = None
    if test_loader is not None:
        final_test = evaluate_model(
            model,
            test_loader,
            [best_threshold],
            deps,
            device,
            dice_loss_weight=args.dice_loss_weight,
        )

    write_history_csv(output_dir / "history.csv", history)
    maybe_write_training_plot(history, output_dir)
    write_prediction_grids(
        model,
        by_split["validation"] + by_split["test"],
        deps,
        args.image_size,
        mean,
        std,
        best_threshold,
        device,
        output_dir,
        args.qualitative_count,
    )
    validation_report = split_metrics_payload(
        "validation", final_validation, best_epoch, best_threshold
    )
    write_key_value_csv(output_dir / "validation_metrics.csv", validation_report)
    test_report = (
        split_metrics_payload("test", final_test, best_epoch, best_threshold)
        if final_test
        else None
    )
    if test_report is not None:
        write_key_value_csv(output_dir / "test_metrics.csv", test_report)

    metrics = {
        "model_name": args.model_name,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dice_loss_weight": args.dice_loss_weight,
        "seed": args.seed,
        "device": device,
        "split_counts": {split: len(items) for split, items in by_split.items()},
        "best_epoch": best_epoch,
        "selected_threshold": best_threshold,
        "train_seconds": train_seconds,
        "val_dice": validation_report["dice"],
        "val_iou": validation_report["iou"],
        "val_pixel_accuracy": validation_report["pixel_accuracy"],
        "val_ms_per_image": validation_report["milliseconds_per_image"],
        "test_dice": test_report["dice"] if test_report else None,
        "test_iou": test_report["iou"] if test_report else None,
        "test_pixel_accuracy": test_report["pixel_accuracy"] if test_report else None,
        "test_ms_per_image": test_report["milliseconds_per_image"]
        if test_report
        else None,
        "validation_metrics_csv": "validation_metrics.csv",
        "test_metrics_csv": "test_metrics.csv" if test_report else None,
        "dataset_provenance": "dataset_provenance.json",
        "environment": "environment.json",
        "validation_threshold_metrics": final_validation["threshold_metrics"],
        "test_threshold_metrics": final_test["threshold_metrics"]
        if final_test
        else None,
    }
    write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = metric_counts_from_binary_masks(
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0],
    )
    sample_pairs = output_dir / "sample_pairs.csv"
    with sample_pairs.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "mask_path", "split"])
        writer.writeheader()
        writer.writerow(
            {"image_path": "images/a.png", "mask_path": "masks/a.png", "split": "train"}
        )
        writer.writerow(
            {
                "image_path": "images/b.png",
                "mask_path": "masks/b.png",
                "split": "validation",
            }
        )
        writer.writerow(
            {"image_path": "images/c.png", "mask_path": "masks/c.png", "split": "test"}
        )
    records = read_pairs_csv(sample_pairs, output_dir)
    dry_run_dir = output_dir / "dry_run"
    command_train(
        argparse.Namespace(
            pairs_csv=str(sample_pairs),
            data_root=str(output_dir),
            output_dir=str(dry_run_dir),
            model_name="nvidia/segformer-b0-finetuned-ade-512-512",
            dataset_name="synthetic smoke segmentation pairs",
            dataset_citation="not applicable",
            dataset_url=None,
            split_rule="fixed synthetic train/validation/test rows",
            image_size=256,
            epochs=1,
            batch_size=1,
            learning_rate=5e-5,
            weight_decay=0.01,
            dice_loss_weight=1.0,
            thresholds=[0.3, 0.5, 0.7],
            seed=7,
            num_workers=0,
            qualitative_count=1,
            save_model=False,
            dry_run=True,
            cpu=False,
            amp=False,
            device="auto",
        )
    )
    summary = {
        "status": "ok",
        "dependency_report": dependency_report(),
        "metric_example": counts_to_metrics(counts),
        "split_counts": {
            key: len(value) for key, value in split_records(records).items()
        },
        "dry_run_artifacts": [
            "config.json",
            "dataset_provenance.json",
            "environment.json",
        ],
        "training_command": [
            sys.executable,
            str(Path(__file__).name),
            "train",
            "--pairs-csv",
            "pairs.csv",
            "--model-name",
            "nvidia/segformer-b0-finetuned-ade-512-512",
            "--image-size",
            "256",
            "--epochs",
            "3",
        ],
    }
    (output_dir / "segformer_training_script_smoke.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"SegFormer training-script smoke check wrote artifacts to {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    deps = subparsers.add_parser("dependency-check")
    deps.add_argument("--allow-missing-deps", action="store_true")
    deps.add_argument("--output-dir")
    deps.set_defaults(func=command_dependency_check)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument(
        "--output-dir",
        default=str(Path(tempfile.gettempdir()) / "adl-segformer-training-smoke"),
    )
    smoke.set_defaults(func=command_smoke)

    train = subparsers.add_parser("train")
    train.add_argument("--pairs-csv", required=True)
    train.add_argument("--data-root")
    train.add_argument("--output-dir", required=True)
    train.add_argument(
        "--model-name", default="nvidia/segformer-b0-finetuned-ade-512-512"
    )
    train.add_argument("--dataset-name", default="Kvasir-SEG")
    train.add_argument("--dataset-citation", default="jha2019kvasir")
    train.add_argument(
        "--dataset-url",
        default="https://datasets.simula.no/kvasir-seg/",
    )
    train.add_argument(
        "--split-rule",
        default="CSV split column; test split is held out until the final report",
    )
    train.add_argument("--image-size", type=int, default=256)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--dice-loss-weight", type=float, default=1.0)
    train.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--qualitative-count", type=int, default=6)
    train.add_argument("--save-model", action="store_true")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    train.add_argument("--cpu", action="store_true", help="Shortcut for --device cpu.")
    train.add_argument("--amp", action="store_true")
    train.set_defaults(func=command_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
