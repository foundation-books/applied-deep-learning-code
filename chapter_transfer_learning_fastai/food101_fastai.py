#!/usr/bin/env python3
"""Train or smoke-test a fast.ai Food-101 transfer-learning classifier.

The default mode uses fast.ai's FOOD dataset helper and does not evaluate the
reserved official test split unless --evaluate-test is passed. The smoke mode
creates a tiny synthetic image directory so CI or a chapter audit can check the
pipeline without downloading Food-101 or pretrained weights.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import random
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_plot_style() -> Any:
    import importlib.util

    for parent in Path(__file__).resolve().parents:
        for candidate in (
            parent / "adl_plot_style.py",
            parent / "book" / "_maintenance" / "plot_style.py",
        ):
            if not candidate.exists():
                continue
            spec = importlib.util.spec_from_file_location("adl_plot_style", candidate)
            if spec is None or spec.loader is None:
                break
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Could not locate adl_plot_style.py")


PLOT_STYLE = load_plot_style()


def public_command() -> str:
    command = " ".join(shlex.quote(part) for part in sys.argv)
    home = str(Path.home())
    if home and home in command:
        command = command.replace(home, "<home>")
    return command


@dataclass
class RunMetadata:
    command: str
    timestamp_utc: str
    seed: int
    mode: str
    data_source: str
    data_root: str
    data_split_rule: str
    reserved_test_rule: str
    architecture: str
    pretrained: bool
    image_size: int
    resize_size: int
    batch_size: int
    epochs: int
    freeze_epochs: int
    base_lr: float
    lr_mult: int
    valid_pct: float
    device: str
    host: str
    python: str
    platform: str


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "not-installed"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def model_parameter_counts(model: Any) -> dict[str, int]:
    parameters = list(model.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
    }


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def import_fastai_stack(allow_missing_deps: bool) -> dict[str, Any] | None:
    try:
        fastai_all = importlib.import_module("fastai.vision.all")
        tv_models = importlib.import_module("torchvision.models")
        torch = importlib.import_module("torch")
        return {"fastai": fastai_all, "torchvision_models": tv_models, "torch": torch}
    except Exception as exc:
        message = f"Could not import fast.ai/PyTorch/TorchVision stack: {exc}"
        if allow_missing_deps:
            print(f"[WARN] {message}")
            return None
        raise RuntimeError(message) from exc


def detect_device(torch: Any) -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def architecture_callable(tv_models: Any, name: str) -> Any:
    mapping = {
        "resnet18": tv_models.resnet18,
        "resnet34": tv_models.resnet34,
        "resnet50": tv_models.resnet50,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported architecture: {name}")
    return mapping[name]


def architecture_weights(tv_models: Any, name: str, pretrained: bool) -> Any:
    if not pretrained:
        return None
    weights_name = {
        "resnet18": "ResNet18_Weights",
        "resnet34": "ResNet34_Weights",
        "resnet50": "ResNet50_Weights",
    }[name]
    weights_enum = getattr(tv_models, weights_name)
    return weights_enum.DEFAULT


def create_synthetic_images(
    root: Path, *, classes: int, items_per_class: int, size: int, seed: int
) -> list[Path]:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(
            f"Pillow is required for synthetic smoke images: {exc}"
        ) from exc

    rng = random.Random(seed)
    image_root = root / "images"
    files: list[Path] = []
    for class_index in range(classes):
        class_name = f"synthetic_food_{class_index:02d}"
        class_dir = image_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        base_color = (
            (53 * (class_index + 1)) % 255,
            (97 * (class_index + 3)) % 255,
            (151 * (class_index + 5)) % 255,
        )
        for item_index in range(items_per_class):
            image = Image.new("RGB", (size, size), base_color)
            draw = ImageDraw.Draw(image)
            for _ in range(5):
                x0 = rng.randint(0, max(0, size - 12))
                y0 = rng.randint(0, max(0, size - 12))
                x1 = min(size, x0 + rng.randint(6, max(7, size // 2)))
                y1 = min(size, y0 + rng.randint(6, max(7, size // 2)))
                color = tuple(
                    (channel + rng.randint(20, 120)) % 255 for channel in base_color
                )
                draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            path = class_dir / f"img_{item_index:03d}.png"
            image.save(path)
            files.append(path)
    return files


def food101_files(fastai: Any, split_name: str, path: Path) -> Any:
    txt = path / f"{split_name}.txt"
    if not txt.exists():
        txt = path / "meta" / f"{split_name}.txt"
    if not txt.exists():
        raise FileNotFoundError(
            f"Could not find Food-101 split file for {split_name!r} under {path}"
        )
    return fastai.L(txt.read_text(encoding="utf-8").splitlines()).map(
        lambda item: path / "images" / f"{item}.jpg"
    )


def save_learner_artifacts(learn: Any, output_dir: Path) -> None:
    recorder = getattr(learn, "recorder", None)
    values = getattr(recorder, "values", None)
    metric_names = getattr(recorder, "metric_names", None)
    if values is not None and metric_names is not None:
        names = [str(name) for name in metric_names]
        rows: list[dict[str, Any]] = []
        for index, raw_row in enumerate(values):
            row_values = list(raw_row)
            if len(names) == len(row_values) + 1 and names[0] == "epoch":
                row_values = [index] + row_values
            if len(names) != len(row_values):
                names = [f"value_{i}" for i in range(len(row_values))]
            rows.append({name: value for name, value in zip(names, row_values)})
        write_csv(output_dir / "training_log.csv", names, rows)
        save_training_curve_plot(output_dir, rows)
    save_lr_schedule_plot(learn, output_dir)
    try:
        learn.export(output_dir / "learner_export.pkl")
    except Exception as exc:
        (output_dir / "export_warning.txt").write_text(
            str(exc) + "\n", encoding="utf-8"
        )


def import_matplotlib(output_dir: Path) -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        (output_dir / "plot_warning.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return None


def import_plotnine(output_dir: Path) -> dict[str, Any] | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import pandas as pd
        import plotnine as p9

        return {
            "pd": pd,
            "aes": p9.aes,
            "facet_wrap": p9.facet_wrap,
            "geom_line": p9.geom_line,
            "geom_point": p9.geom_point,
            "ggplot": p9.ggplot,
            "labs": p9.labs,
            "p9": p9,
            "scale_color_manual": p9.scale_color_manual,
            "style": PLOT_STYLE,
            "theme_minimal": p9.theme_minimal,
        }
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        (output_dir / "plot_warning.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return None


def _is_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def save_training_curve_plot(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    pn = import_plotnine(output_dir)
    if pn is None or not rows:
        return

    numeric_names = [
        name
        for name in rows[0].keys()
        if name != "time" and all(_is_float(row.get(name)) for row in rows)
    ]
    loss_names = [name for name in numeric_names if "loss" in name]
    other_metric_names = [
        name for name in numeric_names if name not in {"epoch"} and "loss" not in name
    ]
    if not loss_names and not other_metric_names:
        return

    xs = [float(row.get("epoch", index)) for index, row in enumerate(rows)]
    records = []
    if loss_names:
        for name in loss_names:
            for x_value, row in zip(xs, rows):
                records.append(
                    {
                        "epoch": x_value,
                        "panel": "loss",
                        "metric": name,
                        "value": float(row[name]),
                    }
                )
    if other_metric_names:
        for name in other_metric_names:
            for x_value, row in zip(xs, rows):
                records.append(
                    {
                        "epoch": x_value,
                        "panel": "metric",
                        "metric": name,
                        "value": float(row[name]),
                    }
                )

    data = pn["pd"].DataFrame.from_records(records)
    data["panel"] = pn["pd"].Categorical(
        data["panel"], categories=["loss", "metric"], ordered=True
    )
    plot = (
        pn["ggplot"](data, pn["aes"]("epoch", "value", color="metric", group="metric"))
        + pn["geom_line"](size=0.8)
        + pn["geom_point"](size=2.0)
        + pn["facet_wrap"]("~panel", scales="free_y", nrow=1)
        + pn["scale_color_manual"](
            values=pn["style"].palette_for(sorted(data["metric"].unique()))
        )
        + pn["labs"](x="epoch", y="metric value", color="metric")
        + pn["style"].plot_theme(pn["p9"])
    )
    panel_count = int(bool(loss_names)) + int(bool(other_metric_names))
    plot.save(
        output_dir / "training_curves.png",
        width=5 * panel_count,
        height=4,
        units="in",
        dpi=PLOT_STYLE.PLOT_DPI,
        verbose=False,
    )


def save_lr_schedule_plot(learn: Any, output_dir: Path) -> None:
    recorder = getattr(learn, "recorder", None)
    if (
        recorder is None
        or not hasattr(recorder, "plot_sched")
        or not hasattr(recorder, "hps")
    ):
        return
    plt = import_matplotlib(output_dir)
    if plt is None:
        return
    try:
        PLOT_STYLE.apply_matplotlib_style(plt)
        recorder.plot_sched()
        figure = plt.gcf()
        figure.tight_layout()
        figure.savefig(
            output_dir / "learning_rate_schedule.png", dpi=PLOT_STYLE.PLOT_DPI
        )
        plt.close(figure)
    except Exception as exc:  # pragma: no cover - depends on fast.ai recorder state
        (output_dir / "lr_schedule_warning.txt").write_text(
            str(exc) + "\n", encoding="utf-8"
        )


def validate_stage(
    learn: Any,
    *,
    stage: str,
    epochs: int,
    training_elapsed: float,
    parameter_counts: dict[str, int],
) -> dict[str, Any]:
    validation_start = time.perf_counter()
    valid_loss, valid_accuracy = learn.validate()
    validation_elapsed = time.perf_counter() - validation_start
    return {
        "stage": stage,
        "epochs": epochs,
        "training_elapsed_seconds": round(training_elapsed, 3),
        "validation_elapsed_seconds": round(validation_elapsed, 3),
        "valid_loss": to_float(valid_loss),
        "valid_accuracy": to_float(valid_accuracy),
        **parameter_counts,
    }


def write_stage_metrics(output_dir: Path, stage_metrics: list[dict[str, Any]]) -> None:
    if not stage_metrics:
        return
    write_csv(
        output_dir / "stage_metrics.csv",
        [
            "stage",
            "epochs",
            "training_elapsed_seconds",
            "validation_elapsed_seconds",
            "valid_loss",
            "valid_accuracy",
            "total_parameters",
            "trainable_parameters",
            "frozen_parameters",
        ],
        stage_metrics,
    )


def save_validation_evidence(
    learn: Any, dls: Any, torch: Any, output_dir: Path, max_top_losses: int
) -> None:
    try:
        preds, targets = learn.get_preds(dl=dls.valid)
    except Exception as exc:  # pragma: no cover - depends on fast.ai learner state
        (output_dir / "evidence_warning.txt").write_text(
            str(exc) + "\n", encoding="utf-8"
        )
        return

    if len(targets) == 0:
        return

    classes = [str(item) for item in dls.vocab]
    pred_idx = preds.argmax(dim=1)
    target_idx = targets.long()
    confusion = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
    for target, predicted in zip(target_idx.cpu().tolist(), pred_idx.cpu().tolist()):
        confusion[target, predicted] += 1

    matrix_rows = []
    for target_index, class_name in enumerate(classes):
        row = {"target": class_name}
        row.update(
            {
                classes[pred_index]: int(confusion[target_index, pred_index])
                for pred_index in range(len(classes))
            }
        )
        matrix_rows.append(row)
    write_csv(
        output_dir / "validation_confusion_matrix.csv",
        ["target"] + classes,
        matrix_rows,
    )

    top_confusions = []
    for target_index, target_name in enumerate(classes):
        for pred_index, pred_name in enumerate(classes):
            if target_index == pred_index:
                continue
            count = int(confusion[target_index, pred_index])
            if count:
                top_confusions.append(
                    {"target": target_name, "predicted": pred_name, "count": count}
                )
    top_confusions.sort(key=lambda row: row["count"], reverse=True)
    write_csv(
        output_dir / "validation_top_confusions.csv",
        ["target", "predicted", "count"],
        top_confusions[:50],
    )

    target_prob = preds[torch.arange(len(target_idx)), target_idx].clamp_min(1e-12)
    top_prob = preds[torch.arange(len(pred_idx)), pred_idx]
    losses = -target_prob.log()
    top_n = min(max_top_losses, len(losses))
    top_indices = torch.argsort(losses, descending=True)[:top_n].cpu().tolist()
    items = list(getattr(dls.valid_ds, "items", []))
    top_loss_rows = []
    for rank, item_index in enumerate(top_indices, start=1):
        target = int(target_idx[item_index])
        predicted = int(pred_idx[item_index])
        item_path = str(items[item_index]) if item_index < len(items) else ""
        top_loss_rows.append(
            {
                "rank": rank,
                "item": item_path,
                "target": classes[target],
                "predicted": classes[predicted],
                "target_probability": round(to_float(target_prob[item_index]), 8),
                "predicted_probability": round(to_float(top_prob[item_index]), 8),
                "loss": round(to_float(losses[item_index]), 8),
            }
        )
    write_csv(
        output_dir / "validation_top_losses.csv",
        [
            "rank",
            "item",
            "target",
            "predicted",
            "target_probability",
            "predicted_probability",
            "loss",
        ],
        top_loss_rows,
    )
    save_top_loss_grid(output_dir / "validation_top_losses.png", top_loss_rows)


def save_top_loss_grid(
    path: Path, rows: list[dict[str, Any]], *, thumb_size: int = 160
) -> None:
    if not rows:
        return
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - optional dependency
        (path.parent / "top_loss_grid_warning.txt").write_text(
            str(exc) + "\n", encoding="utf-8"
        )
        return

    cells = []
    for row in rows[:16]:
        item = Path(str(row["item"]))
        if not item.exists():
            continue
        try:
            image = Image.open(item).convert("RGB")
            image.thumbnail((thumb_size, thumb_size))
            canvas = Image.new("RGB", (thumb_size, thumb_size + 48), "white")
            x = (thumb_size - image.width) // 2
            y = (thumb_size - image.height) // 2
            canvas.paste(image, (x, y))
            draw = ImageDraw.Draw(canvas)
            label = f"{row['rank']}. {row['target']} -> {row['predicted']}"
            draw.text((4, thumb_size + 4), label[:34], fill="black")
            draw.text((4, thumb_size + 24), f"loss {row['loss']}", fill="black")
            cells.append(canvas)
        except Exception:
            continue

    if not cells:
        return
    columns = min(4, len(cells))
    rows_needed = (len(cells) + columns - 1) // columns
    grid = Image.new(
        "RGB", (columns * thumb_size, rows_needed * (thumb_size + 48)), "white"
    )
    for index, cell in enumerate(cells):
        grid.paste(
            cell,
            ((index % columns) * thumb_size, (index // columns) * (thumb_size + 48)),
        )
    grid.save(path)


def write_result_summary(
    output_dir: Path, metadata: RunMetadata, metrics: dict[str, Any]
) -> None:
    write_csv(
        output_dir / "result_summary.csv",
        [
            "mode",
            "architecture",
            "pretrained",
            "image_size",
            "resize_size",
            "batch_size",
            "freeze_epochs",
            "epochs",
            "base_lr",
            "lr_mult",
            "host",
            "platform",
            "device",
            "total_parameters",
            "trainable_parameters_after_build",
            "trainable_parameters_final",
            "valid_accuracy",
            "test_accuracy",
            "elapsed_seconds",
        ],
        [
            {
                "mode": metadata.mode,
                "architecture": metadata.architecture,
                "pretrained": metadata.pretrained,
                "image_size": metadata.image_size,
                "resize_size": metadata.resize_size,
                "batch_size": metadata.batch_size,
                "freeze_epochs": metadata.freeze_epochs,
                "epochs": metadata.epochs,
                "base_lr": metadata.base_lr,
                "lr_mult": metadata.lr_mult,
                "host": metadata.host,
                "platform": metadata.platform,
                "device": metadata.device,
                "total_parameters": metrics.get("total_parameters", ""),
                "trainable_parameters_after_build": metrics.get(
                    "trainable_parameters_after_build", ""
                ),
                "trainable_parameters_final": metrics.get(
                    "trainable_parameters_final", ""
                ),
                "valid_accuracy": metrics.get("valid_accuracy", ""),
                "test_accuracy": metrics.get("test_accuracy", ""),
                "elapsed_seconds": metrics.get("elapsed_seconds", ""),
            }
        ],
    )


def run(args: argparse.Namespace) -> int:
    stack = import_fastai_stack(args.allow_missing_deps)
    versions = {
        "fastai": package_version("fastai"),
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "timm": package_version("timm"),
        "pillow": package_version("pillow"),
    }

    if args.check_deps:
        status = "dependency-check-passed" if stack is not None else "missing dependencies"
        print(json.dumps({"status": status, "versions": versions}, indent=2, sort_keys=True))
        return 0 if stack is not None or args.allow_missing_deps else 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(
        " ".join(shlex.quote(part) for part in sys.argv) + "\n", encoding="utf-8"
    )
    write_json(output_dir / "versions.json", versions)

    if stack is None:
        write_json(
            output_dir / "metrics.json",
            {"status": "skipped", "reason": "missing dependencies"},
        )
        return 0 if args.allow_missing_deps else 1

    fastai = stack["fastai"]
    tv_models = stack["torchvision_models"]
    torch = stack["torch"]
    set_reproducible_seed(args.seed)

    if args.smoke and not args.smoke_pretrained and not args.no_pretrained:
        args.no_pretrained = True
        print(
            "[INFO] Smoke mode disables pretrained weights by default; pass --smoke-pretrained to allow weight downloads."
        )

    mode = "smoke-synthetic" if args.smoke else "food101"
    device = detect_device(torch)
    pretrained = not args.no_pretrained
    metadata_base: dict[str, Any] = {
        "command": public_command(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "mode": mode,
        "architecture": args.architecture,
        "pretrained": pretrained,
        "image_size": args.image_size,
        "resize_size": args.resize_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "freeze_epochs": args.freeze_epochs,
        "base_lr": args.base_lr,
        "lr_mult": args.lr_mult,
        "valid_pct": args.valid_pct,
        "device": device,
        "host": "not-recorded",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    start = time.perf_counter()
    if args.smoke:
        synthetic_root = output_dir / "synthetic_data"
        items = fastai.L(
            create_synthetic_images(
                synthetic_root,
                classes=args.smoke_classes,
                items_per_class=args.smoke_items_per_class,
                size=max(args.resize_size, args.image_size),
                seed=args.seed,
            )
        )
        test_items = fastai.L([])
        data_source = "synthetic smoke images generated by create_synthetic_images"
        data_root = str(synthetic_root)
        data_split_rule = (
            f"RandomSplitter(valid_pct={args.valid_pct}, seed={args.seed}) over "
            f"{args.smoke_classes} synthetic classes with {args.smoke_items_per_class} items per class"
        )
        reserved_test_rule = (
            "none; smoke mode does not create a reserved official test split"
        )
    else:
        path = fastai.untar_data(fastai.URLs.FOOD)
        items = food101_files(fastai, "train", Path(path))
        test_items = food101_files(fastai, "test", Path(path))
        data_source = "fastai.URLs.FOOD"
        data_root = str(Path(path))
        data_split_rule = (
            f"Food-101 train.txt loaded from {Path(path) / 'meta' / 'train.txt'} or {Path(path) / 'train.txt'}; "
            f"RandomSplitter(valid_pct={args.valid_pct}, seed={args.seed}) creates validation from official train"
        )
        reserved_test_rule = (
            f"Food-101 test.txt loaded from {Path(path) / 'meta' / 'test.txt'} or {Path(path) / 'test.txt'}; "
            "reserved for --evaluate-test after model selection"
        )

    metadata = RunMetadata(
        **metadata_base,
        data_source=data_source,
        data_root=data_root,
        data_split_rule=data_split_rule,
        reserved_test_rule=reserved_test_rule,
    )
    write_json(output_dir / "metadata.json", asdict(metadata))

    food = fastai.DataBlock(
        blocks=(fastai.ImageBlock, fastai.CategoryBlock),
        get_y=fastai.parent_label,
        splitter=fastai.RandomSplitter(valid_pct=args.valid_pct, seed=args.seed),
        item_tfms=fastai.Resize(args.resize_size),
        batch_tfms=fastai.aug_transforms(size=args.image_size)
        if not args.no_aug
        else [],
    )
    dls = food.dataloaders(items, bs=args.batch_size, num_workers=args.num_workers)
    arch = architecture_callable(tv_models, args.architecture)
    weights = architecture_weights(tv_models, args.architecture, pretrained)
    learn = fastai.vision_learner(
        dls,
        arch,
        pretrained=pretrained,
        weights=weights,
        metrics=fastai.accuracy,
    )
    build_counts = model_parameter_counts(learn.model)

    metrics: dict[str, Any] = {
        "status": "built",
        "mode": mode,
        "data_source": metadata.data_source,
        "data_root": metadata.data_root,
        "data_split_rule": metadata.data_split_rule,
        "reserved_test_rule": metadata.reserved_test_rule,
        "classes": list(dls.vocab),
        "n_train_items": len(dls.train_ds),
        "n_valid_items": len(dls.valid_ds),
        "n_reserved_test_items": len(test_items),
        "total_parameters": build_counts["total_parameters"],
        "trainable_parameters_after_build": build_counts["trainable_parameters"],
        "frozen_parameters_after_build": build_counts["frozen_parameters"],
    }

    if args.freeze_epochs > 0 or args.epochs > 0:
        stage_metrics: list[dict[str, Any]] = []
        if args.freeze_epochs > 0:
            learn.freeze()
            frozen_counts = model_parameter_counts(learn.model)
            stage_start = time.perf_counter()
            learn.fit_one_cycle(args.freeze_epochs, slice(args.base_lr), pct_start=0.99)
            stage_metrics.append(
                validate_stage(
                    learn,
                    stage="frozen",
                    epochs=args.freeze_epochs,
                    training_elapsed=time.perf_counter() - stage_start,
                    parameter_counts=frozen_counts,
                )
            )

        if args.epochs > 0:
            learn.unfreeze()
            unfrozen_counts = model_parameter_counts(learn.model)
            unfrozen_base_lr = args.base_lr / 2
            stage_start = time.perf_counter()
            learn.fit_one_cycle(
                args.epochs,
                slice(unfrozen_base_lr / args.lr_mult, unfrozen_base_lr),
                pct_start=args.pct_start,
                div=args.div,
            )
            stage_metrics.append(
                validate_stage(
                    learn,
                    stage="unfrozen",
                    epochs=args.epochs,
                    training_elapsed=time.perf_counter() - stage_start,
                    parameter_counts=unfrozen_counts,
                )
            )

        if stage_metrics:
            metrics.update(
                {
                    "valid_loss": stage_metrics[-1]["valid_loss"],
                    "valid_accuracy": stage_metrics[-1]["valid_accuracy"],
                    "stage_metrics": stage_metrics,
                }
            )
            write_stage_metrics(output_dir, stage_metrics)
        save_learner_artifacts(learn, output_dir)
        if not args.skip_evidence_artifacts:
            save_validation_evidence(learn, dls, torch, output_dir, args.max_top_losses)
    else:
        metrics.update(
            {"training": "skipped", "reason": "epochs and freeze_epochs are both zero"}
        )

    final_counts = model_parameter_counts(learn.model)
    metrics.update(
        {
            "trainable_parameters_final": final_counts["trainable_parameters"],
            "frozen_parameters_final": final_counts["frozen_parameters"],
        }
    )

    if args.evaluate_test and len(test_items) > 0:
        test_dl = dls.test_dl(test_items, with_labels=True)
        test_loss, test_accuracy = learn.validate(dl=test_dl)
        metrics.update(
            {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}
        )
    elif args.evaluate_test:
        metrics.update(
            {
                "test_evaluation": "skipped",
                "reason": "no reserved test items in smoke mode",
            }
        )
    else:
        metrics.update({"test_evaluation": "not requested"})

    metrics["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    write_result_summary(output_dir, metadata, metrics)
    write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=["resnet18", "resnet34", "resnet50"],
        default="resnet34",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=460)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--epochs", type=int, default=5, help="Unfrozen fine-tuning epochs."
    )
    parser.add_argument(
        "--freeze-epochs", type=int, default=1, help="Frozen-head warm-up epochs."
    )
    parser.add_argument("--base-lr", type=float, default=3e-3)
    parser.add_argument(
        "--lr-mult",
        type=int,
        default=100,
        help="Discriminative LR ratio for unfrozen fine-tuning.",
    )
    parser.add_argument(
        "--pct-start",
        type=float,
        default=0.3,
        help="One-cycle warm-up fraction for the unfrozen stage.",
    )
    parser.add_argument(
        "--div",
        type=float,
        default=5.0,
        help="One-cycle initial LR divisor for the unfrozen stage.",
    )
    parser.add_argument("--valid-pct", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs/food101-fastai")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate reserved official test split once after model selection.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained weights; useful for offline smoke tests.",
    )
    parser.add_argument(
        "--smoke-pretrained",
        action="store_true",
        help="Allow pretrained weights in smoke mode; may download weights.",
    )
    parser.add_argument(
        "--no-aug", action="store_true", help="Disable random augmentation."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny synthetic images instead of downloading Food-101.",
    )
    parser.add_argument("--smoke-classes", type=int, default=2)
    parser.add_argument("--smoke-items-per-class", type=int, default=6)
    parser.add_argument("--max-top-losses", type=int, default=16)
    parser.add_argument("--skip-evidence-artifacts", action="store_true")
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--allow-missing-deps", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    supplied_args = sys.argv[1:] if argv is None else argv
    if not supplied_args:
        parser.print_help()
        print(
            "\nNo run started. Pass --smoke for a synthetic smoke test or provide "
            "explicit Food-101 training options when the dataset download and "
            "training time are intended."
        )
        return 0
    args = parser.parse_args(argv)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
