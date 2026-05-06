#!/usr/bin/env python3
"""Train the MNIST MLP from the chapter with direct PyTorch."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path


INSTALL_HELP = """Install PyTorch and torchvision before training:
  poetry install --with pytorch
"""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the chapter MNIST MLP using direct PyTorch.",
    )
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory used by torchvision to cache MNIST.",
    )
    parser.add_argument(
        "--limit-train",
        type=positive_int,
        help="Use only the first N training examples.",
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
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker process count.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run one epoch on a small subset for environment checks.",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check whether PyTorch and torchvision can import.",
    )
    parser.add_argument(
        "--allow-missing-deps",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def apply_quick_defaults(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.epochs = 1
    args.limit_train = min(args.limit_train or 1024, 1024)
    args.limit_val = min(args.limit_val or 256, 256)
    args.limit_test = min(args.limit_test or 256, 256)


def import_torch_stack():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Subset, random_split
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - depends on local packages.
        raise RuntimeError(f"Could not import PyTorch stack: {exc}\n{INSTALL_HELP}") from exc

    return torch, nn, DataLoader, Subset, random_split, datasets, transforms


def limit_dataset(dataset, limit: int | None, subset_type):
    if limit is None:
        return dataset
    return subset_type(dataset, range(min(limit, len(dataset))))


def choose_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    args = parse_args()
    apply_quick_defaults(args)

    try:
        torch, nn, DataLoader, Subset, random_split, datasets, transforms = (
            import_torch_stack()
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 0 if args.check_deps and args.allow_missing_deps else 1

    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("torchvision dependency check: ok")

    if args.check_deps:
        return 0

    torch.manual_seed(args.seed)
    device = choose_device(torch)
    print("device:", device)

    transform = transforms.ToTensor()
    train_full = datasets.MNIST(
        root=str(args.data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=str(args.data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    train_set, val_set = random_split(
        train_full,
        [50000, 10000],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_set = limit_dataset(train_set, args.limit_train, Subset)
    val_set = limit_dataset(val_set, args.limit_val, Subset)
    test_set = limit_dataset(test_set, args.limit_test, Subset)

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    images, labels = next(iter(train_loader))
    print("batch images:", images.shape)
    print("batch labels:", labels.shape)

    class MNISTMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.net(x)

    model = MNISTMLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train_one_epoch(loader):
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

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / total, correct / total

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(train_loader)
        val_loss, val_acc = evaluate(val_loader)
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    test_loss, test_acc = evaluate(test_loader)
    elapsed = time.perf_counter() - start
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"test_loss={test_loss:.4f} test_accuracy={test_acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
