#!/usr/bin/env python3
"""Direct PyTorch ResNet-family CIFAR-10 training script."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import math
import platform
import shlex
import sys
import time
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
CHAPTER_DIR = CODE_DIR
DEFAULT_FIGURE_DIR = CHAPTER_DIR / "artifacts" / "figures"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

MODEL_CONFIGS = {
    "resnet18": ("basic", [2, 2, 2, 2]),
    "resnet34": ("basic", [3, 4, 6, 3]),
    "resnet50": ("bottleneck", [3, 4, 6, 3]),
    "tiny-resnet": ("basic", [1, 1, 1, 1]),
}


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


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or parsed >= 1:
        raise argparse.ArgumentTypeError("must be in the range [0, 1)")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ResNet-family PyTorch model on CIFAR-10."
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Optional artifact prefix for saved files. When omitted, the legacy "
            "<model>-cifar10-pytorch prefix is used."
        ),
    )
    parser.add_argument("--epochs", type=positive_int, default=50)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--validation-size", type=positive_int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--limit-train",
        type=positive_int,
        help="Use only the first N training examples after the validation split.",
    )
    parser.add_argument(
        "--limit-val",
        type=positive_int,
        help="Use only the first N validation examples.",
    )
    parser.add_argument(
        "--limit-test",
        type=positive_int,
        help="Use only the first N test examples.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=CODE_DIR / "data",
        help="Directory used by torchvision to cache CIFAR-10.",
    )
    parser.add_argument(
        "--stem",
        choices=["cifar", "imagenet"],
        default="cifar",
    )
    parser.add_argument(
        "--model-variant",
        choices=list(MODEL_CONFIGS),
        default="resnet18",
        help="Choose a named ResNet variant. ResNet18 is the default student-scale model.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adamw"],
        default="sgd",
    )
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=nonnegative_float, default=5e-4)
    parser.add_argument("--schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--warmup-epochs", type=nonnegative_int, default=0)
    parser.add_argument(
        "--normalization", choices=["batchnorm", "none"], default="batchnorm"
    )
    parser.add_argument("--dropout-rate", type=probability, default=0.0)
    parser.add_argument("--augmentation", choices=["none", "basic"], default="basic")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use synthetic data, tiny model, small splits, and one epoch.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the held-out test set. Use only for a final selected or verification run.",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save history figures, confusion matrix, per-class accuracy, and metadata.",
    )
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument(
        "--allow-missing-deps",
        action="store_true",
        help="Return success with a warning when optional ML dependencies are absent.",
    )
    return parser.parse_args(argv)


def command_line(argv: list[str]) -> str:
    return " ".join(
        shlex.quote(part) for part in ["python", Path(__file__).name, *argv]
    )


def artifact_prefix(args: argparse.Namespace) -> str:
    if args.run_id is None:
        return f"{args.model_variant}-cifar10-pytorch"
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in args.run_id.strip()
    ).strip("-_")
    if not safe:
        raise SystemExit(
            "--run-id must contain at least one letter, number, hyphen, or underscore"
        )
    return safe


def package_version(name: str) -> str:
    try:
        return f"{name} {version(name)}"
    except PackageNotFoundError:
        return f"{name} not installed"


def import_torch(allow_missing_deps: bool):
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset, Subset
    except Exception as exc:  # pragma: no cover
        message = f"Could not import PyTorch dependencies: {exc}"
        if allow_missing_deps:
            print(f"[WARN] {message}")
            return None
        raise SystemExit(message) from exc
    return torch, nn, DataLoader, Dataset, Subset


def import_torchvision(allow_missing_deps: bool):
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        message = f"Could not import torchvision dependencies: {exc}"
        if allow_missing_deps:
            print(f"[WARN] {message}")
            return None
        raise SystemExit(message) from exc
    return datasets, transforms


def apply_quick_defaults(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.epochs = 1
    args.validation_size = min(args.validation_size, 64)
    args.synthetic_data = True
    if args.model_variant != "tiny-resnet":
        args.model_variant = "tiny-resnet"


def choose_device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_summary(torch: Any, device: Any) -> str:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps is not None and mps.is_available())
    return f"device={device}; cuda={torch.cuda.is_available()}; mps={mps_available}"


def make_model_classes(nn: Any):
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(
            self,
            in_channels: int,
            filters: int,
            stride: int = 1,
            *,
            norm_layer: Any,
            use_conv_bias: bool,
        ):
            super().__init__()
            out_channels = filters * self.expansion
            self.conv1 = nn.Conv2d(
                in_channels,
                filters,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=use_conv_bias,
            )
            self.bn1 = norm_layer(filters)
            self.conv2 = nn.Conv2d(
                filters,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=use_conv_bias,
            )
            self.bn2 = norm_layer(out_channels)
            self.relu = nn.ReLU(inplace=True)
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=use_conv_bias,
                    ),
                    norm_layer(out_channels),
                )
            else:
                self.downsample = None

        def forward(self, x: Any) -> Any:
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(identity)
            out = self.relu(out + identity)
            return out

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(
            self,
            in_channels: int,
            filters: int,
            stride: int = 1,
            *,
            norm_layer: Any,
            use_conv_bias: bool,
        ):
            super().__init__()
            out_channels = filters * self.expansion
            self.conv1 = nn.Conv2d(
                in_channels, filters, kernel_size=1, bias=use_conv_bias
            )
            self.bn1 = norm_layer(filters)
            self.conv2 = nn.Conv2d(
                filters,
                filters,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=use_conv_bias,
            )
            self.bn2 = norm_layer(filters)
            self.conv3 = nn.Conv2d(
                filters, out_channels, kernel_size=1, bias=use_conv_bias
            )
            self.bn3 = norm_layer(out_channels)
            self.relu = nn.ReLU(inplace=True)
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=use_conv_bias,
                    ),
                    norm_layer(out_channels),
                )
            else:
                self.downsample = None

        def forward(self, x: Any) -> Any:
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.downsample is not None:
                identity = self.downsample(identity)
            out = self.relu(out + identity)
            return out

    class CIFARResNet(nn.Module):
        def __init__(
            self,
            block: Any,
            block_counts: list[int],
            stem: str,
            normalization: str,
            dropout_rate: float,
        ):
            super().__init__()
            self.in_channels = 64
            norm_layer = (
                nn.BatchNorm2d
                if normalization == "batchnorm"
                else lambda _channels: nn.Identity()
            )
            use_conv_bias = normalization == "none"
            self.norm_layer = norm_layer
            self.use_conv_bias = use_conv_bias
            self.dropout_rate = dropout_rate
            if stem == "imagenet":
                self.stem = nn.Sequential(
                    nn.Conv2d(
                        3, 64, kernel_size=7, stride=2, padding=3, bias=use_conv_bias
                    ),
                    norm_layer(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
            else:
                self.stem = nn.Sequential(
                    nn.Conv2d(
                        3, 64, kernel_size=3, stride=1, padding=1, bias=use_conv_bias
                    ),
                    norm_layer(64),
                    nn.ReLU(inplace=True),
                )
            self.layer1 = self._make_stage(block, 64, block_counts[0], stride=1)
            self.layer2 = self._make_stage(block, 128, block_counts[1], stride=2)
            self.layer3 = self._make_stage(block, 256, block_counts[2], stride=2)
            self.layer4 = self._make_stage(block, 512, block_counts[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.dropout = (
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            )
            self.fc = nn.Linear(512 * block.expansion, 10)

        def _make_stage(
            self, block: Any, filters: int, blocks: int, stride: int
        ) -> Any:
            layers = [
                block(
                    self.in_channels,
                    filters,
                    stride,
                    norm_layer=self.norm_layer,
                    use_conv_bias=self.use_conv_bias,
                )
            ]
            self.in_channels = filters * block.expansion
            for _ in range(1, blocks):
                layers.append(
                    block(
                        self.in_channels,
                        filters,
                        1,
                        norm_layer=self.norm_layer,
                        use_conv_bias=self.use_conv_bias,
                    )
                )
            return nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.dropout(x)
            return self.fc(x)

    return BasicBlock, Bottleneck, CIFARResNet


def build_model(nn: Any, args: argparse.Namespace) -> Any:
    BasicBlock, Bottleneck, CIFARResNet = make_model_classes(nn)
    block_type, block_counts = MODEL_CONFIGS[args.model_variant]
    block = Bottleneck if block_type == "bottleneck" else BasicBlock
    normalization = getattr(args, "normalization", "batchnorm")
    dropout_rate = getattr(args, "dropout_rate", 0.0)
    return CIFARResNet(block, block_counts, args.stem, normalization, dropout_rate)


def parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def make_synthetic_datasets(torch: Any, Dataset: Any, args: argparse.Namespace):
    class SyntheticCIFAR10(Dataset):
        def __init__(self, count: int, seed: int):
            generator = torch.Generator().manual_seed(seed)
            self.images = torch.rand(count, 3, 32, 32, generator=generator)
            self.labels = torch.randint(
                0, len(CLASS_NAMES), (count,), generator=generator
            )

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int):
            return self.images[index], self.labels[index]

    train_count = 128 if args.quick else 4096
    val_count = 64 if args.quick else args.validation_size
    test_count = 64 if args.quick else 1000
    print("Dataset source: synthetic CIFAR-10-shaped data")
    return (
        SyntheticCIFAR10(train_count, args.seed),
        SyntheticCIFAR10(val_count, args.seed + 1),
        SyntheticCIFAR10(test_count, args.seed + 2),
    )


def make_real_datasets(torch: Any, Subset: Any, args: argparse.Namespace):
    stack = import_torchvision(args.allow_missing_deps)
    if stack is None:
        return None
    datasets, transforms = stack
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )
    train_transforms: list[Any] = []
    if args.augmentation == "basic":
        train_transforms.extend(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        )
    train_transform = transforms.Compose(
        [*train_transforms, transforms.ToTensor(), normalize]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_augmented = datasets.CIFAR10(
        root=str(args.data_dir), train=True, download=True, transform=train_transform
    )
    train_eval = datasets.CIFAR10(
        root=str(args.data_dir), train=True, download=True, transform=eval_transform
    )
    test_set = datasets.CIFAR10(
        root=str(args.data_dir), train=False, download=True, transform=eval_transform
    )

    if args.validation_size >= len(train_eval):
        raise SystemExit(
            "--validation-size must be smaller than the CIFAR-10 training set"
        )
    indices = torch.randperm(
        len(train_eval), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    val_indices = indices[: args.validation_size]
    train_indices = indices[args.validation_size :]
    if args.limit_train is not None:
        train_indices = train_indices[: args.limit_train]
    if args.limit_val is not None:
        val_indices = val_indices[: args.limit_val]
    if args.limit_test is not None:
        test_set = Subset(test_set, list(range(min(args.limit_test, len(test_set)))))
    print("Dataset source: torchvision.datasets.CIFAR10")
    return (
        Subset(train_augmented, train_indices),
        Subset(train_eval, val_indices),
        test_set,
    )


def build_optimizer_and_scheduler(torch: Any, model: Any, args: argparse.Namespace):
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if args.schedule == "cosine" and args.warmup_epochs > 0:
        if args.warmup_epochs >= args.epochs:
            raise SystemExit("--warmup-epochs must be smaller than --epochs")

        def lr_factor(epoch_index: int) -> float:
            if epoch_index < args.warmup_epochs:
                return (epoch_index + 1) / args.warmup_epochs
            decay_steps = max(1, args.epochs - args.warmup_epochs)
            progress = min(1.0, (epoch_index - args.warmup_epochs) / decay_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    elif args.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    else:
        scheduler = None
    return optimizer, scheduler


def train_one_epoch(
    torch: Any, model: Any, loader: Any, device: Any, loss_fn: Any, optimizer: Any
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(
    torch: Any, model: Any, loader: Any, device: Any, loss_fn: Any
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            predicted = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            y_true.extend(int(label) for label in labels.cpu())
            y_pred.extend(int(label) for label in predicted.cpu())
    return total_loss / total, correct / total, y_true, y_pred


def save_history(
    rows: list[dict[str, float | int]], output_dir: Path, prefix: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}-history.csv"
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def import_plotnine() -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import pandas as pd
        import plotnine as p9
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Saving figures requires plotnine. Install with poetry install --with figures"
        ) from exc
    return {
        "pd": pd,
        "aes": p9.aes,
        "coord_fixed": p9.coord_fixed,
        "element_text": p9.element_text,
        "facet_wrap": p9.facet_wrap,
        "geom_col": p9.geom_col,
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
        "scale_fill_manual": p9.scale_fill_manual,
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
) -> Path:
    plot.save(path, width=width, height=height, units="in", dpi=dpi, verbose=False)
    return path


def row_metric_values(rows: list[dict[str, float | int]], *names: str) -> list[float]:
    for name in names:
        values = [float(row[name]) for row in rows if name in row]
        if values:
            return values
    return []


def save_training_curves(
    rows: list[dict[str, float | int]], output_dir: Path, prefix: str
) -> Path:
    pn = import_plotnine()
    path = output_dir / f"{prefix}-training-curves.png"
    if len(rows) == 1:
        row = rows[0]
        records = []
        for panel, values in [
            ("Loss after epoch 1", [float(row["train_loss"]), float(row["val_loss"])]),
            (
                "Accuracy after epoch 1",
                [float(row["train_accuracy"]), float(row["val_accuracy"])],
            ),
        ]:
            offset = max(values) * 0.045 if values else 0.02
            for split, value in zip(["train", "validation"], values):
                records.append(
                    {
                        "panel": panel,
                        "split": split,
                        "value": value,
                        "label": f"{value:.3f}",
                        "label_y": value + offset,
                    }
                )
        data = pn["pd"].DataFrame.from_records(records)
        data["panel"] = pn["pd"].Categorical(
            data["panel"],
            categories=["Loss after epoch 1", "Accuracy after epoch 1"],
            ordered=True,
        )
        plot = (
            pn["ggplot"](data, pn["aes"]("split", "value", fill="split"))
            + pn["geom_col"](width=0.55)
            + pn["geom_text"](pn["aes"](y="label_y", label="label"), size=8)
            + pn["facet_wrap"]("~panel", scales="free_y", nrow=1)
            + pn["scale_fill_manual"](
                values=pn["style"].palette_for(["train", "validation"])
            )
            + pn["labs"](
                title="CIFAR-10 ResNet one-epoch verification", x="", y="metric value"
            )
            + pn["style"].plot_theme(pn["p9"])
            + pn["theme"](legend_position="none")
        )
        return save_plot(plot, path, width=7.8, height=3.4)

    epochs = [int(row["epoch"]) for row in rows]
    records = []
    for panel, split, values in [
        ("Loss", "train", row_metric_values(rows, "train_loss", "loss")),
        ("Loss", "validation", row_metric_values(rows, "val_loss")),
        ("Accuracy", "train", row_metric_values(rows, "train_accuracy", "accuracy")),
        ("Accuracy", "validation", row_metric_values(rows, "val_accuracy")),
    ]:
        for epoch, value in zip(epochs, values):
            records.append(
                {"epoch": epoch, "panel": panel, "split": split, "value": value}
            )
    data = pn["pd"].DataFrame.from_records(records)
    data["panel"] = pn["pd"].Categorical(
        data["panel"], categories=["Loss", "Accuracy"], ordered=True
    )
    plot = (
        pn["ggplot"](data, pn["aes"]("epoch", "value", color="split", group="split"))
        + pn["geom_line"](size=0.8)
        + pn["geom_point"](size=2.2)
        + pn["facet_wrap"]("~panel", scales="free_y", nrow=1)
        + pn["scale_color_manual"](
            values=pn["style"].palette_for(["train", "validation"])
        )
        + pn["scale_x_continuous"](breaks=epochs)
        + pn["labs"](
            title="CIFAR-10 ResNet training", x="epoch", y="metric value", color="split"
        )
        + pn["style"].plot_theme(pn["p9"])
    )
    return save_plot(plot, path, width=8.8, height=3.4)


def confusion_matrix(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1
    return matrix


def save_confusion_csv(matrix: list[list[int]], output_dir: Path, prefix: str) -> Path:
    path = output_dir / f"{prefix}-confusion-matrix.csv"
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(["true_label", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *row])
    return path


def save_per_class_accuracy(
    matrix: list[list[int]], output_dir: Path, prefix: str
) -> Path:
    path = output_dir / f"{prefix}-per-class-accuracy.csv"
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(["class", "correct", "total", "accuracy"])
        for row_index, class_name in enumerate(CLASS_NAMES):
            total = sum(matrix[row_index])
            correct = matrix[row_index][row_index]
            writer.writerow(
                [
                    class_name,
                    correct,
                    total,
                    f"{(correct / total if total else 0.0):.6f}",
                ]
            )
    return path


def save_confusion_png(matrix: list[list[int]], output_dir: Path, prefix: str) -> Path:
    pn = import_plotnine()
    path = output_dir / f"{prefix}-confusion-matrix.png"
    largest = max(max(row) for row in matrix) or 1
    records = []
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            records.append(
                {
                    "true_label": CLASS_NAMES[row_index],
                    "predicted_label": CLASS_NAMES[col_index],
                    "count": value,
                    "label": str(value),
                    "text_color": "white" if value > largest * 0.55 else "#222222",
                }
            )
    plot = (
        pn["ggplot"](
            pn["pd"].DataFrame.from_records(records),
            pn["aes"]("predicted_label", "true_label", fill="count"),
        )
        + pn["geom_tile"](color="white", size=0.35)
        + pn["geom_text"](pn["aes"](label="label", color="text_color"), size=6)
        + pn["scale_color_identity"]()
        + pn["scale_fill_gradient"](
            low=pn["style"].CONFUSION_LOW, high=pn["style"].CONFUSION_HIGH
        )
        + pn["scale_x_discrete"](limits=CLASS_NAMES)
        + pn["scale_y_discrete"](limits=list(reversed(CLASS_NAMES)))
        + pn["coord_fixed"]()
        + pn["labs"](
            title="CIFAR-10 validation-set confusion matrix",
            x="predicted label",
            y="true label",
            fill="examples",
        )
        + pn["style"].plot_theme(pn["p9"])
        + pn["theme"](axis_text_x=pn["element_text"](rotation=45, ha="right"))
    )
    return save_plot(plot, path, width=6.4, height=5.8)


def write_metadata(
    output_dir: Path,
    prefix: str,
    files: list[Path],
    *,
    args: argparse.Namespace,
    command: str,
    run_started_utc: str,
    torch_version: str,
    torchvision_version: str,
    device_info: str,
    parameters: int,
    train_count: int,
    val_count: int,
    test_count: int | None,
    elapsed: float,
    val_loss: float,
    val_acc: float,
    best_val_acc: float,
    test_loss: float | None,
    test_acc: float | None,
) -> Path:
    path = output_dir / f"{prefix}-metadata.txt"
    path.write_text(
        "\n".join(
            [
                "CNN architectures ResNet CIFAR-10 PyTorch experiment",
                f"Generated by {Path(__file__).resolve().relative_to(REPO_ROOT)}",
                f"Run started UTC: {run_started_utc}",
                f"Run ID: {args.run_id if args.run_id is not None else 'default'}",
                f"Artifact prefix: {prefix}",
                f"Artifact directory: {output_dir}",
                f"Command: {command}",
                f"Python: {platform.python_version()}",
                f"Matplotlib: {package_version('matplotlib')}",
                f"Plotnine: {package_version('plotnine')}",
                f"PyTorch: {torch_version}",
                f"torchvision: {torchvision_version}",
                f"Device summary: {device_info}",
                "Host: not-recorded",
                f"Platform: {platform.platform()}",
                f"Seed: {args.seed}",
                f"Model variant: {args.model_variant}",
                f"Stem: {args.stem}",
                f"Optimizer: {args.optimizer}",
                f"Learning rate: {args.learning_rate}",
                f"Momentum: {args.momentum if args.optimizer == 'sgd' else 'not used'}",
                f"Schedule: {args.schedule}",
                f"Warmup epochs: {args.warmup_epochs}",
                f"Weight decay: {args.weight_decay}",
                f"Normalization: {args.normalization}",
                f"Conv bias policy: {'enabled without normalization' if args.normalization == 'none' else 'disabled with batch normalization'}",
                f"Dropout rate: {args.dropout_rate}",
                f"Batch size: {args.batch_size}",
                f"Epochs: {args.epochs}",
                f"Augmentation: {args.augmentation}",
                f"Validation size: {args.validation_size}",
                f"Train limit: {args.limit_train}",
                f"Validation limit: {args.limit_val}",
                f"Test limit: {args.limit_test}",
                f"Train examples: {train_count}",
                f"Validation examples: {val_count}",
                f"Test examples: {test_count if test_count is not None else 'not evaluated'}",
                f"Parameter count: {parameters}",
                f"Elapsed seconds: {elapsed:.2f}",
                f"Final validation loss: {val_loss:.4f}",
                f"Final validation accuracy: {val_acc:.4f}",
                f"Best validation accuracy: {best_val_acc:.4f}",
                f"Test evaluation: {'enabled' if test_acc is not None else 'skipped'}",
                *(
                    [
                        f"Test loss: {test_loss:.4f}",
                        f"Test accuracy: {test_acc:.4f}",
                    ]
                    if test_loss is not None and test_acc is not None
                    else []
                ),
                "Generated files:",
                *[f"- {file.name}" for file in files],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    apply_quick_defaults(args)
    stack = import_torch(args.allow_missing_deps)
    if stack is None:
        return 0
    torch, nn, DataLoader, Dataset, Subset = stack
    run_started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    command = command_line(argv)

    torch.manual_seed(args.seed)
    device = choose_device(torch)
    device_info = device_summary(torch, device)
    torchvision_version = package_version("torchvision")
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("torchvision:", torchvision_version)
    print("Device summary:", device_info)
    print("Command:", command)
    print("Run started UTC:", run_started_utc)
    print(f"model_variant={args.model_variant}")
    print(f"stem={args.stem}")
    print(f"optimizer={args.optimizer}")
    print(f"schedule={args.schedule}")
    print(f"warmup_epochs={args.warmup_epochs}")
    print(f"normalization={args.normalization}")
    print(f"dropout_rate={args.dropout_rate}")
    print(f"augmentation={args.augmentation}")

    if args.check_deps:
        if not args.synthetic_data:
            import_torchvision(args.allow_missing_deps)
        print("Dependency check passed.")
        return 0

    if args.synthetic_data:
        train_set, val_set, test_set = make_synthetic_datasets(torch, Dataset, args)
    else:
        datasets = make_real_datasets(torch, Subset, args)
        if datasets is None:
            return 0
        train_set, val_set, test_set = datasets

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_loader = (
        DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)
        if args.evaluate_test
        else None
    )
    first_images, first_labels = next(iter(train_loader))
    print("train:", (len(train_set), *tuple(first_images.shape[1:])), (len(train_set),))
    print("val:", len(val_set))
    if args.evaluate_test:
        print("test:", len(test_set))
    else:
        print("test: skipped (--evaluate-test not set)")
    print("batch images:", tuple(first_images.shape))
    print("batch labels:", tuple(first_labels.shape))
    print("seed:", args.seed)
    print("batch_size:", args.batch_size)
    print("epochs:", args.epochs)
    print("validation_size:", args.validation_size)

    model = build_model(nn, args).to(device)
    parameters = parameter_count(model)
    print(f"parameter_count={parameters}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimizer_and_scheduler(torch, model, args)

    rows: list[dict[str, float | int]] = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss, train_acc = train_one_epoch(
            torch, model, train_loader, device, loss_fn, optimizer
        )
        val_loss, val_acc, _, _ = evaluate(torch, model, val_loader, device, loss_fn)
        if scheduler is not None:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": learning_rate,
        }
        rows.append(row)
        print(
            f"epoch={epoch} "
            f"lr={learning_rate:.6g} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    elapsed = time.perf_counter() - start
    val_loss, val_acc, val_true, val_pred = evaluate(
        torch, model, val_loader, device, loss_fn
    )
    best_val_acc = max(float(row["val_accuracy"]) for row in rows)
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"final_val_loss={val_loss:.4f} final_val_accuracy={val_acc:.4f}")
    print(f"best_val_accuracy={best_val_acc:.4f}")
    test_loss: float | None = None
    test_acc: float | None = None
    if test_loader is not None:
        test_loss, test_acc, _, _ = evaluate(torch, model, test_loader, device, loss_fn)
        print(f"test_loss={test_loss:.4f} test_accuracy={test_acc:.4f}")
    else:
        print("test_evaluation=skipped")
    print("validation predicted:", [CLASS_NAMES[index] for index in val_pred[:8]])
    print("validation true:", [CLASS_NAMES[index] for index in val_true[:8]])

    if args.save_figures:
        prefix = artifact_prefix(args)
        args.figure_dir.mkdir(parents=True, exist_ok=True)
        matrix = confusion_matrix(val_true, val_pred)
        written = [
            save_history(rows, args.figure_dir, prefix),
            save_training_curves(rows, args.figure_dir, prefix),
            save_confusion_csv(matrix, args.figure_dir, prefix),
            save_confusion_png(matrix, args.figure_dir, prefix),
            save_per_class_accuracy(matrix, args.figure_dir, prefix),
        ]
        written.append(
            write_metadata(
                args.figure_dir,
                prefix,
                written,
                args=args,
                command=command,
                run_started_utc=run_started_utc,
                torch_version=torch.__version__,
                torchvision_version=torchvision_version,
                device_info=device_info,
                parameters=parameters,
                train_count=len(train_set),
                val_count=len(val_set),
                test_count=len(test_set) if args.evaluate_test else None,
                elapsed=elapsed,
                val_loss=val_loss,
                val_acc=val_acc,
                best_val_acc=best_val_acc,
                test_loss=test_loss,
                test_acc=test_acc,
            )
        )
        for path in written:
            print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
