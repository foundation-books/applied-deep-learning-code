#!/usr/bin/env python3
"""ConvNeXt Food-101 fine-tuning companion script."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any


INSTALL_HELP = """Install the shared PyTorch environment before running:
  cd chapter_cnn_revisited
  poetry install --with transfer-learning
"""

CSV_LINETERMINATOR = "\n"


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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def valid_pct(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune or smoke-test a TorchVision ConvNeXt on Food-101."
    )
    parser.add_argument("--data-root", type=Path, help="Food-101 root directory.")
    parser.add_argument(
        "--model-variant",
        choices=["convnext_tiny", "convnext_small", "convnext_base"],
        default="convnext_tiny",
    )
    parser.add_argument("--image-size", type=positive_int, default=224)
    parser.add_argument("--resize-size", type=positive_int, default=256)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--valid-pct", type=valid_pct, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-epochs", type=nonnegative_int, default=1)
    parser.add_argument("--epochs", type=nonnegative_int, default=4)
    parser.add_argument("--head-lr", type=positive_float, default=3e-4)
    parser.add_argument("--fine-tune-lr", type=positive_float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument("--limit-train", type=positive_int)
    parser.add_argument("--limit-val", type=positive_int)
    parser.add_argument("--limit-test", type=positive_int)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--prediction-examples", type=nonnegative_int, default=64)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download Food-101 with TorchVision if it is not already present.",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        help="TorchVision download root; defaults to the parent of --data-root or ~/.cache/torch/datasets.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save training-curve and confusion-matrix PNG artifacts when matplotlib is installed.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--device", help="Override device, e.g. cpu, cuda, or mps.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/convnext-food101")
    )
    parser.add_argument("--smoke", action="store_true", help="Use synthetic images.")
    parser.add_argument("--quick", action="store_true", help="Tiny synthetic run.")
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument(
        "--allow-missing-deps",
        action="store_true",
        help="Return success when optional PyTorch dependencies are missing.",
    )
    return parser.parse_args(argv)


def apply_quick_defaults(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.smoke = True
    args.no_pretrained = True
    args.image_size = min(args.image_size, 64)
    args.resize_size = max(args.image_size, min(args.resize_size, 72))
    args.batch_size = min(args.batch_size, 4)
    args.freeze_epochs = 1
    args.epochs = 0
    args.limit_train = min(args.limit_train or 8, 8)
    args.limit_val = min(args.limit_val or 4, 4)


def import_torch_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torchvision
        from torchvision import transforms
    except Exception as exc:  # pragma: no cover - optional dependency guard.
        raise RuntimeError(
            f"Could not import PyTorch/TorchVision: {exc}\n{INSTALL_HELP}"
        ) from exc
    return torch, torchvision, transforms


def print_dependency_summary(torch: Any, torchvision: Any) -> None:
    print(f"python={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"torch={torch.__version__}")
    print(f"torchvision={torchvision.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(
        f"mps_available={getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}"
    )


def resolve_device(torch: Any, requested: str | None) -> Any:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def candidate_food101_roots() -> list[Path]:
    return [
        Path.cwd() / "food-101",
        Path.cwd().parent / "food-101",
        Path.home() / ".fastai" / "data" / "food-101",
        Path.home() / ".cache" / "torch" / "datasets" / "food-101",
    ]


def split_file(root: Path, split_name: str) -> Path:
    for candidate in [root / f"{split_name}.txt", root / "meta" / f"{split_name}.txt"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {split_name}.txt under {root}.")


def resolve_data_root(data_root: Path | None) -> Path:
    if data_root is not None:
        root = data_root.expanduser().resolve()
        if not (root / "images").is_dir():
            raise FileNotFoundError(f"{root} does not contain images/.")
        split_file(root, "train")
        split_file(root, "test")
        return root
    for root in candidate_food101_roots():
        if (root / "images").is_dir():
            try:
                split_file(root, "train")
                split_file(root, "test")
            except FileNotFoundError:
                continue
            return root.resolve()
    checked = "\n  ".join(str(path) for path in candidate_food101_roots())
    raise FileNotFoundError(
        f"Food-101 not found. Pass --data-root. Checked:\n  {checked}"
    )


def default_download_root(data_root: Path | None) -> Path:
    if data_root is not None:
        expanded = data_root.expanduser()
        return expanded.parent if expanded.name == "food-101" else expanded
    return Path.home() / ".cache" / "torch" / "datasets"


def ensure_food101_download(
    torchvision: Any, *, data_root: Path | None, download_root: Path | None
) -> Path:
    root = (download_root or default_download_root(data_root)).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    datasets = torchvision.datasets
    datasets.Food101(root=str(root), split="train", download=True)
    datasets.Food101(root=str(root), split="test", download=True)

    extracted = root / "food-101"
    candidates = []
    if data_root is not None:
        candidates.append(data_root.expanduser().resolve())
    candidates.extend([extracted, root])
    for candidate in candidates:
        if (candidate / "images").is_dir():
            split_file(candidate, "train")
            split_file(candidate, "test")
            return candidate.resolve()
    raise FileNotFoundError(
        f"TorchVision download finished, but Food-101 was not found under {root}."
    )


def read_split_entries(root: Path, split_name: str) -> list[str]:
    text = split_file(root, split_name).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def class_name(entry: str) -> str:
    return entry.split("/", maxsplit=1)[0]


def load_food101_records(
    root: Path,
    *,
    seed: int,
    valid_pct_value: float,
    limit_train: int | None,
    limit_val: int | None,
    limit_test: int | None,
) -> tuple[
    list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]], list[str]
]:
    train_entries = read_split_entries(root, "train")
    test_entries = read_split_entries(root, "test")
    class_names = sorted(
        {class_name(entry) for entry in [*train_entries, *test_entries]}
    )
    label_to_index = {name: index for index, name in enumerate(class_names)}

    rng = random.Random(seed)
    indices = list(range(len(train_entries)))
    rng.shuffle(indices)
    valid_count = max(1, int(round(len(indices) * valid_pct_value)))
    valid_indices = indices[:valid_count]
    train_indices = indices[valid_count:]
    if limit_train is not None:
        train_indices = train_indices[:limit_train]
    if limit_val is not None:
        valid_indices = valid_indices[:limit_val]

    def record(entry: str) -> tuple[Path, int]:
        return root / "images" / f"{entry}.jpg", label_to_index[class_name(entry)]

    train_records = [record(train_entries[index]) for index in train_indices]
    val_records = [record(train_entries[index]) for index in valid_indices]
    test_source = test_entries[: limit_test or len(test_entries)]
    test_records = [record(entry) for entry in test_source]
    return train_records, val_records, test_records, class_names


class Food101ImageDataset:
    def __init__(self, records: list[tuple[Path, int]], transform: Any) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        from PIL import Image

        path, label = self.records[index]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return self.transform(rgb), label, str(path)


class SyntheticFood101Dataset:
    def __init__(
        self, torch: Any, *, size: int, image_size: int, num_classes: int, seed: int
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(size, 3, image_size, image_size, generator=generator)
        self.labels = torch.arange(size) % num_classes

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        return self.images[index], int(self.labels[index]), f"synthetic_{index:05d}.jpg"


def make_transforms(
    transforms: Any, *, image_size: int, resize_size: int
) -> tuple[Any, Any]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def build_model(
    torch: Any, torchvision: Any, *, variant: str, pretrained: bool, num_classes: int
) -> tuple[Any, str]:
    models = torchvision.models
    builders = {
        "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights),
        "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights),
        "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights),
    }
    builder, weights_class = builders[variant]
    weights = weights_class.DEFAULT if pretrained else None
    model = builder(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    weights_name = str(weights) if weights is not None else "random initialization"
    return model, weights_name


def set_trainable(model: Any, *, train_body: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = train_body
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def make_loader(
    torch: Any, dataset: Any, *, batch_size: int, shuffle: bool, num_workers: int
) -> Any:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def unpack_batch(batch: Any) -> tuple[Any, Any, Any | None]:
    if len(batch) == 3:
        images, labels, paths = batch
        return images, labels, paths
    images, labels = batch
    return images, labels, None


def autocast_context(torch: Any, device: Any, *, mixed_precision: bool) -> Any:
    if mixed_precision and device.type == "cuda":
        return torch.autocast(device_type=device.type)
    return nullcontext()


@dataclass
class EpochMetrics:
    stage: str
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    elapsed_seconds: float


@dataclass
class DetailedMetrics:
    loss: float
    accuracy: float
    top5_accuracy: float


def run_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    *,
    device: Any,
    criterion: Any,
    optimizer: Any | None,
    mixed_precision: bool,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        images, labels, _paths = unpack_batch(batch)
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with autocast_context(torch, device, mixed_precision=mixed_precision):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
        total += batch_size
    return total_loss / max(total, 1), correct / max(total, 1)


def save_checkpoint(
    torch: Any, path: Path, model: Any, metadata: dict[str, Any]
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, path)


def train_stage(
    torch: Any,
    model: Any,
    *,
    stage: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    train_loader: Any,
    val_loader: Any,
    device: Any,
    mixed_precision: bool,
    checkpoint_dir: Path,
    best_state: dict[str, Any],
) -> list[EpochMetrics]:
    if epochs == 0:
        return []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    metrics: list[EpochMetrics] = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, train_acc = run_epoch(
            torch,
            model,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            mixed_precision=mixed_precision,
        )
        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                torch,
                model,
                val_loader,
                device=device,
                criterion=criterion,
                optimizer=None,
                mixed_precision=mixed_precision,
            )
        elapsed = time.perf_counter() - started
        metrics.append(
            EpochMetrics(
                stage=stage,
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                elapsed_seconds=elapsed,
            )
        )
        print(
            f"{stage} epoch {epoch}: "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} elapsed={elapsed:.2f}s"
        )
        if val_acc > best_state.get("val_accuracy", -1.0):
            checkpoint_path = checkpoint_dir / "best_model.pt"
            checkpoint_metadata = {
                "stage": stage,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
            save_checkpoint(torch, checkpoint_path, model, checkpoint_metadata)
            best_state.update({**checkpoint_metadata, "path": str(checkpoint_path)})
    return metrics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def public_string(value: object) -> str:
    text = str(value)
    home = str(Path.home())
    return text.replace(home, "<home>") if home else text


def public_command() -> str:
    return " ".join(public_string(item) for item in sys.argv)


def write_history(path: Path, rows: list[EpochMetrics]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(rows[0]).keys()) if rows else ["stage"],
            lineterminator=CSV_LINETERMINATOR,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_confusion_matrix(
    path: Path, class_names: list[str], matrix: list[list[int]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator=CSV_LINETERMINATOR)
        writer.writerow(["true_label", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])


def write_per_class_accuracy(
    path: Path, class_names: list[str], matrix: list[list[int]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_name", "correct", "total", "accuracy"],
            lineterminator=CSV_LINETERMINATOR,
        )
        writer.writeheader()
        for index, class_name in enumerate(class_names):
            row = matrix[index]
            correct = row[index]
            total = sum(row)
            accuracy = correct / total if total else 0.0
            writer.writerow(
                {
                    "class_name": class_name,
                    "correct": correct,
                    "total": total,
                    "accuracy": f"{accuracy:.6f}",
                }
            )


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["path", "true_label", "predicted_label", "correct", "confidence"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator=CSV_LINETERMINATOR
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def import_plotnine() -> dict[str, Any] | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import pandas as pd
        import plotnine as p9
    except Exception as exc:  # pragma: no cover - optional plotting dependency.
        print(
            f"Skipping plots because plotnine could not be imported: {exc}",
            file=sys.stderr,
        )
        return None
    return {
        "pd": pd,
        "aes": p9.aes,
        "coord_fixed": p9.coord_fixed,
        "element_text": p9.element_text,
        "facet_wrap": p9.facet_wrap,
        "geom_line": p9.geom_line,
        "geom_point": p9.geom_point,
        "geom_text": p9.geom_text,
        "geom_tile": p9.geom_tile,
        "ggplot": p9.ggplot,
        "labs": p9.labs,
        "p9": p9,
        "scale_color_identity": p9.scale_color_identity,
        "scale_color_manual": p9.scale_color_manual,
        "scale_fill_gradient": p9.scale_fill_gradient,
        "scale_x_continuous": p9.scale_x_continuous,
        "scale_x_discrete": p9.scale_x_discrete,
        "scale_y_discrete": p9.scale_y_discrete,
        "style": PLOT_STYLE,
        "theme": p9.theme,
        "theme_minimal": p9.theme_minimal,
    }


def save_plot(
    plot: Any,
    path: Path,
    *,
    width: float,
    height: float,
    dpi: int = PLOT_STYLE.PLOT_DPI,
) -> None:
    plot.save(path, width=width, height=height, units="in", dpi=dpi, verbose=False)


def write_history_plot(path: Path, rows: list[EpochMetrics]) -> bool:
    if not rows:
        return False
    pn = import_plotnine()
    if pn is None:
        return False
    steps = list(range(1, len(rows) + 1))
    labels = [f"{row.stage}-{row.epoch}" for row in rows]
    records = []
    for panel, split, values in [
        ("Accuracy", "train", [row.train_accuracy for row in rows]),
        ("Accuracy", "validation", [row.val_accuracy for row in rows]),
        ("Loss", "train", [row.train_loss for row in rows]),
        ("Loss", "validation", [row.val_loss for row in rows]),
    ]:
        for step, label, value in zip(steps, labels, values):
            records.append(
                {
                    "step": step,
                    "epoch_label": label,
                    "panel": panel,
                    "split": split,
                    "value": value,
                }
            )
    data = pn["pd"].DataFrame.from_records(records)
    data["panel"] = pn["pd"].Categorical(
        data["panel"], categories=["Accuracy", "Loss"], ordered=True
    )
    plot = (
        pn["ggplot"](data, pn["aes"]("step", "value", color="split", group="split"))
        + pn["geom_line"](size=0.8)
        + pn["geom_point"](size=2.2)
        + pn["facet_wrap"]("~panel", scales="free_y", nrow=1)
        + pn["scale_color_manual"](
            values=pn["style"].palette_for(["train", "validation"])
        )
        + pn["scale_x_continuous"](breaks=steps, labels=labels)
        + pn["labs"](x="logged epoch", y="metric value", color="split")
        + pn["style"].plot_theme(pn["p9"])
        + pn["theme"](axis_text_x=pn["element_text"](rotation=45, ha="right", size=8))
    )
    save_plot(plot, path, width=10.0, height=4.0)
    return True


def write_confusion_plot(
    path: Path,
    class_names: list[str],
    matrix: list[list[int]],
    *,
    max_classes: int = 20,
) -> bool:
    if not matrix:
        return False
    pn = import_plotnine()
    if pn is None:
        return False
    num_classes = len(class_names)
    if num_classes > max_classes:
        totals = []
        for index in range(num_classes):
            true_total = sum(matrix[index])
            predicted_total = sum(row[index] for row in matrix)
            totals.append((true_total + predicted_total, index))
        selected = sorted(
            index for _total, index in sorted(totals, reverse=True)[:max_classes]
        )
        title_suffix = f"top {len(selected)} active classes"
    else:
        selected = list(range(num_classes))
        title_suffix = "all classes"
    reduced = [[matrix[row][col] for col in selected] for row in selected]
    labels = [class_names[index] for index in selected]
    records = []
    largest = max(max(row) for row in reduced) or 1
    for row_index, row in enumerate(reduced):
        for col_index, value in enumerate(row):
            records.append(
                {
                    "true_label": labels[row_index],
                    "predicted_label": labels[col_index],
                    "count": value,
                    "label": str(value) if len(labels) <= 15 else "",
                    "text_color": "white" if value > largest * 0.55 else "#222222",
                }
            )
    plot = (
        pn["ggplot"](
            pn["pd"].DataFrame.from_records(records),
            pn["aes"]("predicted_label", "true_label", fill="count"),
        )
        + pn["geom_tile"](color="white", size=0.2)
        + pn["geom_text"](pn["aes"](label="label", color="text_color"), size=5.8)
        + pn["scale_color_identity"]()
        + pn["scale_fill_gradient"](
            low=pn["style"].CONFUSION_LOW, high=pn["style"].CONFUSION_HIGH
        )
        + pn["scale_x_discrete"](limits=labels)
        + pn["scale_y_discrete"](limits=list(reversed(labels)))
        + pn["coord_fixed"]()
        + pn["labs"](
            title=f"Confusion matrix ({title_suffix})",
            x="Predicted class",
            y="True class",
            fill="examples",
        )
        + pn["style"].plot_theme(pn["p9"])
        + pn["theme"](
            axis_text_x=pn["element_text"](rotation=45, ha="right", size=6.2),
            axis_text_y=pn["element_text"](size=6.2),
        )
    )
    save_plot(plot, path, width=8.0, height=7.0)
    return True


def evaluate_detailed(
    torch: Any,
    model: Any,
    loader: Any,
    *,
    device: Any,
    criterion: Any,
    mixed_precision: bool,
    class_names: list[str],
    prediction_examples: int,
) -> tuple[DetailedMetrics, list[list[int]], list[dict[str, Any]]]:
    model.eval()
    num_classes = len(class_names)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0
    correct = 0
    top5_correct = 0
    total = 0
    examples: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images, labels, paths = unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)
            with autocast_context(torch, device, mixed_precision=mixed_precision):
                logits = model(images)
                loss = criterion(logits, labels)
            batch_size = int(labels.shape[0])
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            topk = logits.topk(k=min(5, num_classes), dim=1).indices
            total_loss += float(loss.detach().cpu()) * batch_size
            correct += int((predicted == labels).sum().detach().cpu())
            top5_correct += int(
                (topk == labels.unsqueeze(1)).any(dim=1).sum().detach().cpu()
            )
            total += batch_size

            labels_cpu = labels.detach().cpu()
            predicted_cpu = predicted.detach().cpu()
            confidence_cpu = confidence.detach().cpu()
            for true_index, predicted_index in zip(
                labels_cpu.tolist(), predicted_cpu.tolist()
            ):
                matrix[true_index, predicted_index] += 1
            if len(examples) < prediction_examples:
                batch_paths = list(paths) if paths is not None else [""] * batch_size
                for item_index in range(batch_size):
                    if len(examples) >= prediction_examples:
                        break
                    true_index = int(labels_cpu[item_index])
                    predicted_index = int(predicted_cpu[item_index])
                    examples.append(
                        {
                            "path": batch_paths[item_index],
                            "true_label": class_names[true_index],
                            "predicted_label": class_names[predicted_index],
                            "correct": true_index == predicted_index,
                            "confidence": f"{float(confidence_cpu[item_index]):.6f}",
                        }
                    )
    metrics = DetailedMetrics(
        loss=total_loss / max(total, 1),
        accuracy=correct / max(total, 1),
        top5_accuracy=top5_correct / max(total, 1),
    )
    return metrics, matrix.cpu().tolist(), examples


def write_detailed_artifacts(
    torch: Any,
    output_dir: Path,
    split_name: str,
    model: Any,
    loader: Any,
    *,
    device: Any,
    mixed_precision: bool,
    class_names: list[str],
    prediction_examples: int,
    save_plots: bool,
) -> dict[str, Any]:
    criterion = torch.nn.CrossEntropyLoss()
    metrics, matrix, examples = evaluate_detailed(
        torch,
        model,
        loader,
        device=device,
        criterion=criterion,
        mixed_precision=mixed_precision,
        class_names=class_names,
        prediction_examples=prediction_examples,
    )
    confusion_path = output_dir / f"{split_name}_confusion_matrix.csv"
    per_class_path = output_dir / f"{split_name}_per_class_accuracy.csv"
    predictions_path = output_dir / f"{split_name}_prediction_examples.csv"
    write_confusion_matrix(confusion_path, class_names, matrix)
    write_per_class_accuracy(per_class_path, class_names, matrix)
    write_predictions(predictions_path, examples)
    artifacts: dict[str, Any] = {
        f"{split_name}_loss": metrics.loss,
        f"{split_name}_accuracy": metrics.accuracy,
        f"{split_name}_top5_accuracy": metrics.top5_accuracy,
        f"{split_name}_confusion_matrix": str(confusion_path),
        f"{split_name}_per_class_accuracy": str(per_class_path),
        f"{split_name}_prediction_examples": str(predictions_path),
    }
    if save_plots:
        confusion_plot_path = output_dir / f"{split_name}_confusion_matrix.png"
        if write_confusion_plot(confusion_plot_path, class_names, matrix):
            artifacts[f"{split_name}_confusion_matrix_plot"] = str(confusion_plot_path)
    return artifacts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_quick_defaults(args)
    try:
        torch, torchvision, transforms = import_torch_stack()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 0 if args.allow_missing_deps else 1

    if args.check_deps:
        print_dependency_summary(torch, torchvision)
        return 0

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = resolve_device(torch, args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained = not args.no_pretrained and not args.smoke
    num_classes = 101
    if args.smoke:
        train_size = args.limit_train or 8
        val_size = args.limit_val or 4
        train_dataset = SyntheticFood101Dataset(
            torch,
            size=train_size,
            image_size=args.image_size,
            num_classes=num_classes,
            seed=args.seed,
        )
        val_dataset = SyntheticFood101Dataset(
            torch,
            size=val_size,
            image_size=args.image_size,
            num_classes=num_classes,
            seed=args.seed + 1,
        )
        test_dataset = None
        class_names = [f"class_{index:03d}" for index in range(num_classes)]
        data_root = "synthetic smoke data"
    else:
        if args.download:
            args.data_root = ensure_food101_download(
                torchvision,
                data_root=args.data_root,
                download_root=args.download_root,
            )
        root = resolve_data_root(args.data_root)
        train_transform, eval_transform = make_transforms(
            transforms, image_size=args.image_size, resize_size=args.resize_size
        )
        train_records, val_records, test_records, class_names = load_food101_records(
            root,
            seed=args.seed,
            valid_pct_value=args.valid_pct,
            limit_train=args.limit_train,
            limit_val=args.limit_val,
            limit_test=args.limit_test,
        )
        train_dataset = Food101ImageDataset(train_records, train_transform)
        val_dataset = Food101ImageDataset(val_records, eval_transform)
        test_dataset = Food101ImageDataset(test_records, eval_transform)
        data_root = str(root)

    train_loader = make_loader(
        torch,
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        torch,
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model, weights_name = build_model(
        torch,
        torchvision,
        variant=args.model_variant,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    model.to(device)

    started = time.perf_counter()
    all_metrics: list[EpochMetrics] = []
    best_state: dict[str, Any] = {}
    set_trainable(model, train_body=False)
    all_metrics.extend(
        train_stage(
            torch,
            model,
            stage="frozen_head",
            epochs=args.freeze_epochs,
            lr=args.head_lr,
            weight_decay=args.weight_decay,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            mixed_precision=args.mixed_precision,
            checkpoint_dir=output_dir,
            best_state=best_state,
        )
    )
    set_trainable(model, train_body=True)
    all_metrics.extend(
        train_stage(
            torch,
            model,
            stage="fine_tune",
            epochs=args.epochs,
            lr=args.fine_tune_lr,
            weight_decay=args.weight_decay,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            mixed_precision=args.mixed_precision,
            checkpoint_dir=output_dir,
            best_state=best_state,
        )
    )
    elapsed = time.perf_counter() - started

    if not best_state:
        final_checkpoint_path = output_dir / "final_model.pt"
        save_checkpoint(
            torch, final_checkpoint_path, model, {"stage": "final", "epoch": 0}
        )
        best_state.update(
            {
                "path": str(final_checkpoint_path),
                "stage": "final",
                "epoch": 0,
                "val_accuracy": None,
            }
        )
    elif Path(best_state["path"]).exists():
        checkpoint = torch.load(best_state["path"], map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    artifact_metrics: dict[str, Any] = {}
    artifact_metrics.update(
        write_detailed_artifacts(
            torch,
            output_dir,
            "validation",
            model,
            val_loader,
            device=device,
            mixed_precision=args.mixed_precision,
            class_names=class_names,
            prediction_examples=args.prediction_examples,
            save_plots=args.save_plots,
        )
    )
    if args.evaluate_test:
        if test_dataset is None:
            print("Skipping test evaluation in smoke mode.")
        else:
            test_loader = make_loader(
                torch,
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            artifact_metrics.update(
                write_detailed_artifacts(
                    torch,
                    output_dir,
                    "test",
                    model,
                    test_loader,
                    device=device,
                    mixed_precision=args.mixed_precision,
                    class_names=class_names,
                    prediction_examples=args.prediction_examples,
                    save_plots=args.save_plots,
                )
            )

    best_val = max((row.val_accuracy for row in all_metrics), default=None)
    metadata = {
        "command": public_command(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "data_root": public_string(data_root),
        "class_count": len(class_names),
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "model_variant": args.model_variant,
        "weights": weights_name,
        "image_size": args.image_size,
        "resize_size": args.resize_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "freeze_epochs": args.freeze_epochs,
        "epochs": args.epochs,
        "head_lr": args.head_lr,
        "fine_tune_lr": args.fine_tune_lr,
        "weight_decay": args.weight_decay,
        "mixed_precision": args.mixed_precision,
        "download": args.download,
        "save_plots": args.save_plots,
        "elapsed_seconds": elapsed,
        "best_checkpoint": best_state,
        "artifacts": {
            key: value
            for key, value in artifact_metrics.items()
            if key.endswith("_confusion_matrix")
            or key.endswith("_per_class_accuracy")
            or key.endswith("_prediction_examples")
            or key.endswith("_confusion_matrix_plot")
        },
    }
    metrics = {
        "best_validation_accuracy": best_val,
        "elapsed_seconds": elapsed,
        **artifact_metrics,
    }
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "metrics.json", metrics)
    write_history(output_dir / "history.csv", all_metrics)
    if args.save_plots:
        history_plot_path = output_dir / "history_plot.png"
        if write_history_plot(history_plot_path, all_metrics):
            metrics["history_plot"] = str(history_plot_path)
            metadata["artifacts"]["history_plot"] = str(history_plot_path)
            write_json(output_dir / "metadata.json", metadata)
            write_json(output_dir / "metrics.json", metrics)
    (output_dir / "command.txt").write_text(public_command() + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
