#!/usr/bin/env python3
"""Pretrained Vision Transformer companion experiment.

The default real-data path fine-tunes a Hugging Face ViT-style checkpoint on a
Food-101 validation split, with CIFAR-10 kept as a smaller fallback. The quick
path uses synthetic images and a tiny random ViT so repository checks do not
download datasets or model weights.
"""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import platform
import random
import shlex
import sys
import time
from typing import Any


DEFAULT_CHECKPOINT = "google/vit-base-patch16-224-in21k"
TINY_RANDOM_CHECKPOINT = "tiny-random-vit"
CSV_LINETERMINATOR = "\n"

INSTALL_HELP = """Install the shared Vision Transformers environment first:
  cd chapter_vision_transformers
  poetry install --with vision-transformers
"""


@dataclass
class DatasetBundle:
    train: Any
    validation: Any
    test: Any | None
    class_names: list[str]
    split_rule: str
    source: str


@dataclass
class PreprocessingInfo:
    image_size: int
    resize_size: int | None
    mean: list[float] | None
    std: list[float] | None
    summary: str


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["food101", "cifar10", "image-folder", "synthetic"],
        default="food101",
        help="Dataset source. image-folder expects class subdirectories or train/val/test folders.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Dataset cache/root. For Food-101 and CIFAR-10 this is the TorchVision cache root.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow TorchVision to download Food-101 or CIFAR-10 if needed.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Hugging Face image-classification checkpoint, or {TINY_RANDOM_CHECKPOINT}.",
    )
    parser.add_argument("--image-size", type=positive_int, default=224)
    parser.add_argument(
        "--resize-size",
        type=positive_int,
        help="Optional resize before center crop. Defaults to direct resize to --image-size.",
    )
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--epochs", type=nonnegative_int, default=1)
    parser.add_argument("--learning-rate", type=positive_float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--valid-pct", type=valid_pct, default=0.2)
    parser.add_argument(
        "--validation-size",
        type=positive_int,
        default=5000,
        help="CIFAR-10 validation size carved out of the official train split.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument("--limit-train", type=positive_int)
    parser.add_argument("--limit-val", type=positive_int)
    parser.add_argument("--limit-test", type=positive_int)
    parser.add_argument(
        "--trainable-mode",
        choices=["head", "last-block", "lora", "full"],
        default="head",
        help="Which pretrained parameters the optimizer may update.",
    )
    parser.add_argument("--lora-r", type=positive_int, default=16)
    parser.add_argument("--lora-alpha", type=positive_int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["query", "value"],
        help="Module names that receive LoRA adapters when --trainable-mode lora is used.",
    )
    parser.add_argument(
        "--lora-modules-to-save",
        nargs="+",
        default=["classifier"],
        help="Non-LoRA modules kept trainable and saved with the adapter.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=positive_int,
        default=1,
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["none", "fp16", "bf16"],
        default="none",
        help="Use fp16 or bf16 autocast where supported.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require checkpoint and processor files to already be cached locally.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the test set. Use only after selecting the recipe from validation evidence.",
    )
    parser.add_argument("--prediction-examples", type=nonnegative_int, default=32)
    parser.add_argument(
        "--save-mistake-gallery",
        action="store_true",
        help="Save a PNG grid of validation mistakes when matplotlib is installed.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the final model state dict as final_model.pt.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pretrained-vit"))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use deterministic synthetic images instead of real data.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny synthetic run for dependency and pipeline checks.",
    )
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument(
        "--allow-missing-deps",
        action="store_true",
        help="Return success when optional ML dependencies are missing.",
    )
    return parser.parse_args(argv)


def make_args(**overrides: Any) -> argparse.Namespace:
    """Return CLI defaults with selected fields overridden for notebooks."""
    args = parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise AttributeError(f"Unknown argument field: {key}")
        setattr(args, key, value)
    return args


def apply_quick_defaults(args: argparse.Namespace) -> None:
    if args.smoke:
        args.dataset = "synthetic"

    if args.dataset == "synthetic" and args.checkpoint == DEFAULT_CHECKPOINT:
        args.checkpoint = TINY_RANDOM_CHECKPOINT

    if not args.quick:
        return

    args.dataset = "synthetic"
    args.checkpoint = TINY_RANDOM_CHECKPOINT
    args.image_size = min(args.image_size, 64)
    args.resize_size = None
    args.batch_size = min(args.batch_size, 4)
    args.epochs = 1
    args.learning_rate = min(args.learning_rate, 5e-4)
    args.trainable_mode = "full"
    args.limit_train = min(args.limit_train or 16, 16)
    args.limit_val = min(args.limit_val or 8, 8)
    args.limit_test = min(args.limit_test or 8, 8)
    args.prediction_examples = min(args.prediction_examples, 8)
    args.local_files_only = True


def load_dependencies(args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        import torch
        import torchvision
        from torchvision import datasets, transforms
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            ViTConfig,
            ViTForImageClassification,
        )
    except Exception as exc:  # pragma: no cover - depends on optional packages.
        message = f"Missing or unusable Vision Transformer dependency: {exc}"
        if args.allow_missing_deps:
            print(message)
            print("Dependency check skipped because --allow-missing-deps was supplied.")
            return None
        raise RuntimeError(f"{message}\n{INSTALL_HELP}") from exc

    peft: Any | None = None
    lora_config: Any | None = None
    get_peft_model: Any | None = None
    peft_error: str | None = None
    if args.check_deps or args.trainable_mode == "lora":
        try:
            import peft
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig
        except Exception as exc:  # pragma: no cover - depends on optional packages.
            peft_error = str(exc)
            if args.trainable_mode == "lora":
                message = f"Missing or unusable PEFT dependency for LoRA: {exc}"
                if args.allow_missing_deps:
                    print(message)
                    print("Dependency check skipped because --allow-missing-deps was supplied.")
                    return None
                raise RuntimeError(f"{message}\n{INSTALL_HELP}") from exc

    return {
        "torch": torch,
        "torchvision": torchvision,
        "datasets": datasets,
        "transforms": transforms,
        "AutoImageProcessor": AutoImageProcessor,
        "AutoModelForImageClassification": AutoModelForImageClassification,
        "ViTConfig": ViTConfig,
        "ViTForImageClassification": ViTForImageClassification,
        "peft": peft,
        "LoraConfig": lora_config,
        "get_peft_model": get_peft_model,
        "peft_error": peft_error,
    }


def print_dependency_summary(deps: dict[str, Any]) -> None:
    torch = deps["torch"]
    torchvision = deps["torchvision"]
    import transformers

    print(f"python={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"torch={torch.__version__}")
    print(f"torchvision={torchvision.__version__}")
    print(f"transformers={transformers.__version__}")
    peft = deps.get("peft")
    if peft is not None:
        print(f"peft={peft.__version__}")
    else:
        print(f"peft=not available ({deps.get('peft_error') or 'not checked'})")
    print(f"cuda_available={torch.cuda.is_available()}")
    mps_backend = getattr(torch.backends, "mps", None)
    print(f"mps_available={bool(mps_backend is not None and mps_backend.is_available())}")


def set_random_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(torch: Any, requested: str | None = None) -> Any:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_summary(torch: Any, device: Any) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        return f"cuda:{torch.cuda.current_device()} {name}"
    if device.type == "mps":
        return "Apple Metal Performance Shaders (mps)"
    return platform.processor() or "cpu"


class IndexedVisionDataset:
    def __init__(
        self,
        base_dataset: Any,
        indices: list[int],
        transform: Any,
        *,
        split_name: str,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        base_index = self.indices[item]
        image, label = self.base_dataset[base_index]
        if self.transform is not None:
            image = self.transform(image)
        return {
            "pixel_values": image,
            "labels": int(label),
            "source": source_id(self.base_dataset, base_index, self.split_name),
        }


class SyntheticVisionDataset:
    def __init__(
        self,
        *,
        length: int,
        num_classes: int,
        image_size: int,
        split_name: str,
        seed: int,
        torch: Any,
    ) -> None:
        self.length = length
        self.num_classes = num_classes
        self.image_size = image_size
        self.split_name = split_name
        self.seed = seed
        self.torch = torch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = self.torch
        label = index % self.num_classes
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.rand(
            3,
            self.image_size,
            self.image_size,
            generator=generator,
            dtype=torch.float32,
        ) * 0.12

        patch = max(4, self.image_size // 5)
        row_slot = label // 2
        col_slot = label % 2
        row = min(self.image_size - patch, 2 + row_slot * (patch + 3))
        col = min(self.image_size - patch, 2 + col_slot * (patch + 3))
        channel = label % 3
        image[channel, row : row + patch, col : col + patch] = 1.0
        image[(channel + 1) % 3, row : row + patch, col : col + patch] = 0.45

        return {
            "pixel_values": image,
            "labels": int(label),
            "source": f"synthetic:{self.split_name}:{index}",
        }


def source_id(base_dataset: Any, index: int, split_name: str) -> str:
    samples = getattr(base_dataset, "samples", None)
    if samples is not None:
        return str(samples[index][0])
    image_files = getattr(base_dataset, "_image_files", None)
    if image_files is not None:
        return str(image_files[index])
    return f"{base_dataset.__class__.__name__}:{split_name}:{index}"


def deterministic_split_indices(
    length: int,
    *,
    validation_count: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, min(validation_count, length - 1))
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    return train_indices, validation_indices


def limit_indices(indices: list[int], limit: int | None) -> list[int]:
    if limit is None:
        return indices
    return indices[:limit]


def build_preprocessing(
    args: argparse.Namespace, deps: dict[str, Any]
) -> tuple[Any, PreprocessingInfo, Any | None]:
    transforms = deps["transforms"]
    if args.checkpoint == TINY_RANDOM_CHECKPOINT:
        info = PreprocessingInfo(
            image_size=args.image_size,
            resize_size=args.resize_size,
            mean=None,
            std=None,
            summary="synthetic tensors in [0, 1]; no pretrained processor",
        )
        return None, info, None

    processor = deps["AutoImageProcessor"].from_pretrained(
        args.checkpoint,
        local_files_only=args.local_files_only,
    )
    mean = list(getattr(processor, "image_mean", [0.5, 0.5, 0.5]))
    std = list(getattr(processor, "image_std", [0.5, 0.5, 0.5]))
    resize_size = args.resize_size or args.image_size

    transform_steps: list[Any] = []
    if args.resize_size:
        transform_steps.append(
            transforms.Resize(
                args.resize_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )
        transform_steps.append(transforms.CenterCrop(args.image_size))
    else:
        transform_steps.append(
            transforms.Resize(
                (args.image_size, args.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )
    transform_steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    info = PreprocessingInfo(
        image_size=args.image_size,
        resize_size=resize_size,
        mean=mean,
        std=std,
        summary=f"Resize/crop to {args.image_size}; normalize with checkpoint image_mean/image_std",
    )
    return transforms.Compose(transform_steps), info, processor


def load_cifar10_bundle(
    args: argparse.Namespace,
    deps: dict[str, Any],
    transform: Any,
) -> DatasetBundle:
    datasets = deps["datasets"]
    root = args.data_root.expanduser()
    train_base = datasets.CIFAR10(root=str(root), train=True, download=args.download)
    test_base = datasets.CIFAR10(root=str(root), train=False, download=args.download)

    train_indices, val_indices = deterministic_split_indices(
        len(train_base),
        validation_count=args.validation_size,
        seed=args.seed,
    )
    train_indices = limit_indices(train_indices, args.limit_train)
    val_indices = limit_indices(val_indices, args.limit_val)
    test_indices = limit_indices(list(range(len(test_base))), args.limit_test)

    return DatasetBundle(
        train=IndexedVisionDataset(train_base, train_indices, transform, split_name="train"),
        validation=IndexedVisionDataset(train_base, val_indices, transform, split_name="validation"),
        test=IndexedVisionDataset(test_base, test_indices, transform, split_name="test"),
        class_names=list(train_base.classes),
        split_rule=(
            f"CIFAR-10 official train split shuffled with seed {args.seed}; "
            f"{len(val_indices)} validation examples held out before training."
        ),
        source=f"TorchVision CIFAR-10 under {root}",
    )


def load_food101_bundle(
    args: argparse.Namespace,
    deps: dict[str, Any],
    transform: Any,
) -> DatasetBundle:
    datasets = deps["datasets"]
    root = args.data_root.expanduser()
    train_base = datasets.Food101(root=str(root), split="train", download=args.download)
    test_base = datasets.Food101(root=str(root), split="test", download=args.download)

    validation_count = max(1, int(round(len(train_base) * args.valid_pct)))
    train_indices, val_indices = deterministic_split_indices(
        len(train_base),
        validation_count=validation_count,
        seed=args.seed,
    )
    train_indices = limit_indices(train_indices, args.limit_train)
    val_indices = limit_indices(val_indices, args.limit_val)
    test_indices = limit_indices(list(range(len(test_base))), args.limit_test)

    return DatasetBundle(
        train=IndexedVisionDataset(train_base, train_indices, transform, split_name="train"),
        validation=IndexedVisionDataset(train_base, val_indices, transform, split_name="validation"),
        test=IndexedVisionDataset(test_base, test_indices, transform, split_name="test"),
        class_names=list(train_base.classes),
        split_rule=(
            f"Food-101 official train split shuffled with seed {args.seed}; "
            f"valid_pct={args.valid_pct}; {len(val_indices)} validation examples held out "
            "before training. Official test split reserved for final evaluation."
        ),
        source=f"TorchVision Food-101 under {root}",
    )


def first_existing_split(root: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def load_image_folder_bundle(
    args: argparse.Namespace,
    deps: dict[str, Any],
    transform: Any,
) -> DatasetBundle:
    datasets = deps["datasets"]
    root = args.data_root.expanduser().resolve()
    train_dir = first_existing_split(root, ["train", "training"])
    val_dir = first_existing_split(root, ["val", "valid", "validation"])
    test_dir = first_existing_split(root, ["test", "testing"])

    if train_dir is None:
        base = datasets.ImageFolder(str(root))
        validation_count = max(1, int(round(len(base) * args.valid_pct)))
        train_indices, val_indices = deterministic_split_indices(
            len(base),
            validation_count=validation_count,
            seed=args.seed,
        )
        train_indices = limit_indices(train_indices, args.limit_train)
        val_indices = limit_indices(val_indices, args.limit_val)
        return DatasetBundle(
            train=IndexedVisionDataset(base, train_indices, transform, split_name="train"),
            validation=IndexedVisionDataset(base, val_indices, transform, split_name="validation"),
            test=None,
            class_names=list(base.classes),
            split_rule=(
                f"ImageFolder root split with seed {args.seed}; "
                f"valid_pct={args.valid_pct}."
            ),
            source=f"ImageFolder at {root}",
        )

    train_base = datasets.ImageFolder(str(train_dir))
    train_indices, val_indices = deterministic_split_indices(
        len(train_base),
        validation_count=max(1, int(round(len(train_base) * args.valid_pct))),
        seed=args.seed,
    )

    if val_dir is not None:
        val_base = datasets.ImageFolder(str(val_dir))
        val_dataset = IndexedVisionDataset(
            val_base,
            limit_indices(list(range(len(val_base))), args.limit_val),
            transform,
            split_name="validation",
        )
        split_rule = f"ImageFolder train directory plus existing validation directory at {val_dir}."
    else:
        train_indices = limit_indices(train_indices, args.limit_train)
        val_indices = limit_indices(val_indices, args.limit_val)
        val_dataset = IndexedVisionDataset(
            train_base,
            val_indices,
            transform,
            split_name="validation",
        )
        split_rule = (
            f"ImageFolder train directory split with seed {args.seed}; "
            f"valid_pct={args.valid_pct}."
        )

    test_dataset = None
    if test_dir is not None:
        test_base = datasets.ImageFolder(str(test_dir))
        test_dataset = IndexedVisionDataset(
            test_base,
            limit_indices(list(range(len(test_base))), args.limit_test),
            transform,
            split_name="test",
        )

    return DatasetBundle(
        train=IndexedVisionDataset(
            train_base,
            limit_indices(train_indices, args.limit_train),
            transform,
            split_name="train",
        ),
        validation=val_dataset,
        test=test_dataset,
        class_names=list(train_base.classes),
        split_rule=split_rule,
        source=f"ImageFolder at {root}",
    )


def load_synthetic_bundle(args: argparse.Namespace, deps: dict[str, Any]) -> DatasetBundle:
    torch = deps["torch"]
    class_count = 4
    train_len = args.limit_train or 64
    val_len = args.limit_val or 32
    test_len = args.limit_test or 32
    class_names = [f"class_{index}" for index in range(class_count)]
    return DatasetBundle(
        train=SyntheticVisionDataset(
            length=train_len,
            num_classes=class_count,
            image_size=args.image_size,
            split_name="train",
            seed=args.seed,
            torch=torch,
        ),
        validation=SyntheticVisionDataset(
            length=val_len,
            num_classes=class_count,
            image_size=args.image_size,
            split_name="validation",
            seed=args.seed + 10_000,
            torch=torch,
        ),
        test=SyntheticVisionDataset(
            length=test_len,
            num_classes=class_count,
            image_size=args.image_size,
            split_name="test",
            seed=args.seed + 20_000,
            torch=torch,
        ),
        class_names=class_names,
        split_rule="Deterministic synthetic train/validation/test tensors.",
        source="Synthetic colored-patch image tensors generated by this script.",
    )


def load_dataset_bundle(
    args: argparse.Namespace,
    deps: dict[str, Any],
    transform: Any,
) -> DatasetBundle:
    if args.dataset == "synthetic":
        return load_synthetic_bundle(args, deps)
    if args.dataset == "food101":
        return load_food101_bundle(args, deps, transform)
    if args.dataset == "cifar10":
        return load_cifar10_bundle(args, deps, transform)
    if args.dataset == "image-folder":
        return load_image_folder_bundle(args, deps, transform)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def collate_examples(batch: list[dict[str, Any]], torch: Any) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in batch]),
        "labels": torch.tensor([example["labels"] for example in batch], dtype=torch.long),
        "sources": [example["source"] for example in batch],
    }


def make_loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    torch: Any,
) -> Any:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_examples(batch, torch),
    )


def build_model(
    args: argparse.Namespace,
    deps: dict[str, Any],
    class_names: list[str],
) -> tuple[Any, str]:
    num_labels = len(class_names)
    id2label = {index: label for index, label in enumerate(class_names)}
    label2id = {label: index for index, label in id2label.items()}

    if args.checkpoint == TINY_RANDOM_CHECKPOINT:
        patch_size = max(4, min(16, args.image_size // 4))
        config = deps["ViTConfig"](
            image_size=args.image_size,
            patch_size=patch_size,
            num_channels=3,
            num_labels=num_labels,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            id2label=id2label,
            label2id=label2id,
        )
        return deps["ViTForImageClassification"](config), TINY_RANDOM_CHECKPOINT

    model = deps["AutoModelForImageClassification"].from_pretrained(
        args.checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        local_files_only=args.local_files_only,
    )
    return model, args.checkpoint


def lora_metadata(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.trainable_mode != "lora":
        return None
    return {
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "target_modules": list(args.lora_target_modules),
        "modules_to_save": list(args.lora_modules_to_save),
    }


def lora_summary(args: argparse.Namespace) -> str:
    metadata = lora_metadata(args)
    if metadata is None:
        return "not used"
    return (
        f"r={metadata['r']}, alpha={metadata['alpha']}, dropout={metadata['dropout']}, "
        f"target_modules={','.join(metadata['target_modules'])}, "
        f"modules_to_save={','.join(metadata['modules_to_save'])}"
    )


def apply_lora(model: Any, args: argparse.Namespace, deps: dict[str, Any]) -> Any:
    if args.trainable_mode != "lora":
        return model

    lora_config_cls = deps.get("LoraConfig")
    get_peft_model = deps.get("get_peft_model")
    if lora_config_cls is None or get_peft_model is None:
        raise RuntimeError("LoRA training requires PEFT. Install with `poetry install --with vision-transformers`.")

    config = lora_config_cls(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=list(args.lora_target_modules),
        lora_dropout=args.lora_dropout,
        bias="none",
        modules_to_save=list(args.lora_modules_to_save),
    )
    return get_peft_model(model, config)


def set_trainable_mode(model: Any, mode: str) -> None:
    if mode == "lora":
        return

    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"

    if mode == "full":
        return

    classifier = getattr(model, "classifier", None)
    if classifier is not None:
        for parameter in classifier.parameters():
            parameter.requires_grad = True

    if mode == "head":
        return

    # Common Hugging Face ViT/DeiT names: vit.encoder.layer[-1] or deit.encoder.layer[-1].
    for backbone_name in ("vit", "deit", "beit"):
        backbone = getattr(model, backbone_name, None)
        encoder = getattr(backbone, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is not None and len(layers) > 0:
            for parameter in layers[-1].parameters():
                parameter.requires_grad = True
        layernorm = getattr(backbone, "layernorm", None)
        if layernorm is not None:
            for parameter in layernorm.parameters():
                parameter.requires_grad = True


def parameter_counts(model: Any) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def autocast_context(torch: Any, device: Any, precision: str) -> Any:
    if precision == "none":
        return nullcontext()
    if device.type == "cuda":
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device.type == "cpu" and precision == "bf16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def make_grad_scaler(torch: Any, device: Any, precision: str) -> Any | None:
    if device.type == "cuda" and precision == "fp16":
        return torch.cuda.amp.GradScaler()
    return None


def train_one_epoch(
    *,
    model: Any,
    loader: Any,
    optimizer: Any,
    device: Any,
    args: argparse.Namespace,
    torch: Any,
    scaler: Any | None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    correct = 0
    seen = 0
    start = time.perf_counter()

    for step, batch in enumerate(loader, start=1):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with autocast_context(torch, device, args.mixed_precision):
            outputs = model(pixel_values=pixel_values, labels=labels)
            raw_loss = outputs.loss
            loss = raw_loss / args.gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = (
            step % args.gradient_accumulation_steps == 0
            or step == len(loader)
        )
        if should_step:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        logits = outputs.logits.detach()
        predictions = logits.argmax(dim=-1)
        batch_size = int(labels.shape[0])
        total_loss += float(raw_loss.detach().cpu()) * batch_size
        correct += int((predictions == labels).sum().detach().cpu())
        seen += batch_size

    elapsed = time.perf_counter() - start
    return {
        "loss": total_loss / max(1, seen),
        "accuracy": correct / max(1, seen),
        "examples": float(seen),
        "elapsed_seconds": elapsed,
        "examples_per_second": seen / elapsed if elapsed > 0 else 0.0,
    }


def evaluate_model(
    *,
    model: Any,
    loader: Any,
    device: Any,
    args: argparse.Namespace,
    torch: Any,
    class_names: list[str],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    num_classes = len(class_names)
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    prediction_rows: list[dict[str, Any]] = []
    gallery_items: list[tuple[Any, int, int, float]] = []
    start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            with autocast_context(torch, device, args.mixed_precision):
                outputs = model(pixel_values=pixel_values, labels=labels)

            probabilities = torch.softmax(outputs.logits.detach(), dim=-1)
            confidences, predictions = probabilities.max(dim=-1)
            batch_size = int(labels.shape[0])
            total_loss += float(outputs.loss.detach().cpu()) * batch_size
            correct += int((predictions == labels).sum().detach().cpu())
            seen += batch_size

            for row_index in range(batch_size):
                true_index = int(labels[row_index].detach().cpu())
                predicted_index = int(predictions[row_index].detach().cpu())
                confidence = float(confidences[row_index].detach().cpu())
                confusion[true_index][predicted_index] += 1

                is_mistake = true_index != predicted_index
                if is_mistake and len(prediction_rows) < args.prediction_examples:
                    prediction_rows.append(
                        {
                            "source": batch["sources"][row_index],
                            "true_label": class_names[true_index],
                            "predicted_label": class_names[predicted_index],
                            "confidence": confidence,
                            "error_note": "",
                        }
                    )
                    gallery_items.append(
                        (
                            batch["pixel_values"][row_index].detach().cpu(),
                            true_index,
                            predicted_index,
                            confidence,
                        )
                    )

    elapsed = time.perf_counter() - start
    return {
        "loss": total_loss / max(1, seen),
        "accuracy": correct / max(1, seen),
        "examples": seen,
        "elapsed_seconds": elapsed,
        "examples_per_second": seen / elapsed if elapsed > 0 else 0.0,
        "confusion_matrix": confusion,
        "prediction_rows": prediction_rows,
        "gallery_items": gallery_items,
    }


def per_class_accuracy(confusion: list[list[int]], class_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(class_names):
        total = sum(confusion[index])
        correct = confusion[index][index]
        rows.append(
            {
                "class_name": name,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }
        )
    return rows


def peak_memory_summary(torch: Any, device: Any) -> str:
    if device.type == "cuda":
        memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        return f"{memory_gb:.3f} GB CUDA max_memory_allocated"
    return "not available for this device"


def command_line(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in ["python", Path(__file__).name, *argv])


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator=CSV_LINETERMINATOR)
        writer.writeheader()
        writer.writerows(history)


def write_confusion_csv(path: Path, confusion: list[list[int]], class_names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator=CSV_LINETERMINATOR)
        writer.writerow(["true_label", *class_names])
        for label, row in zip(class_names, confusion):
            writer.writerow([label, *row])


def write_dict_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator=CSV_LINETERMINATOR)
        writer.writeheader()
        writer.writerows(rows)


def tensor_to_display_image(tensor: Any) -> Any:
    image = tensor.float().clone()
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    return image.permute(1, 2, 0).numpy()


def save_mistake_gallery(
    path: Path,
    gallery_items: list[tuple[Any, int, int, float]],
    class_names: list[str],
) -> None:
    if not gallery_items:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency.
        raise RuntimeError(
            "Saving a mistake gallery requires matplotlib. Install with "
            "`poetry install --with vision-transformers`."
        ) from exc

    columns = min(4, len(gallery_items))
    rows = (len(gallery_items) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.4 * rows))
    if rows == 1 and columns == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.flat)

    for axis, (image, true_index, predicted_index, confidence) in zip(axes_list, gallery_items):
        axis.imshow(tensor_to_display_image(image))
        axis.set_title(
            f"{class_names[true_index]} -> {class_names[predicted_index]}\n{confidence:.2f}",
            fontsize=9,
        )
        axis.axis("off")

    for axis in axes_list[len(gallery_items) :]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def strip_eval_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"confusion_matrix", "prediction_rows", "gallery_items"}
    }


def write_result_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    dataset: DatasetBundle,
    preprocessing: PreprocessingInfo,
    checkpoint_name: str,
    total_parameters: int,
    trainable_parameters: int,
    metrics: dict[str, Any],
    artifact_paths: dict[str, str],
) -> None:
    fieldnames = [
        "dataset_source",
        "class_count",
        "split_rule",
        "checkpoint",
        "preprocessing",
        "trainable_parameters",
        "trainable_mode",
        "lora_config",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "gradient_accumulation_steps",
        "precision",
        "hardware",
        "elapsed_seconds",
        "examples_per_second",
        "peak_memory",
        "validation_accuracy",
        "final_test_accuracy",
        "mistake_table",
        "error_analysis",
    ]
    row = {
        "dataset_source": dataset.source,
        "class_count": len(dataset.class_names),
        "split_rule": dataset.split_rule,
        "checkpoint": checkpoint_name,
        "preprocessing": preprocessing.summary,
        "trainable_parameters": trainable_parameters,
        "trainable_mode": args.trainable_mode,
        "lora_config": lora_summary(args),
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "precision": args.mixed_precision,
        "hardware": metrics["hardware"],
        "elapsed_seconds": metrics["elapsed_seconds"],
        "examples_per_second": metrics["training_examples_per_second"],
        "peak_memory": metrics["peak_memory"],
        "validation_accuracy": metrics["validation"]["accuracy"],
        "final_test_accuracy": metrics["test"].get("accuracy", "not requested"),
        "mistake_table": artifact_paths.get("validation_prediction_examples", ""),
        "error_analysis": "Fill this in after inspecting the mistake table/gallery.",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator=CSV_LINETERMINATOR)
        writer.writeheader()
        writer.writerow(row)

    _ = total_parameters  # Kept explicit in metadata; avoids hiding the homework field.


def run_experiment(
    args: argparse.Namespace,
    *,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    apply_quick_defaults(args)
    deps = load_dependencies(args)
    if deps is None:
        return {"status": "missing-dependencies"}

    if args.check_deps:
        print_dependency_summary(deps)
        return {"status": "dependency-check"}

    torch = deps["torch"]
    set_random_seed(args.seed, torch)
    device = resolve_device(torch)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    transform, preprocessing, _processor = build_preprocessing(args, deps)
    dataset = load_dataset_bundle(args, deps, transform)

    train_loader = make_loader(
        dataset.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        torch=torch,
    )
    validation_loader = make_loader(
        dataset.validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        torch=torch,
    )
    test_loader = None
    if args.evaluate_test:
        if dataset.test is None:
            raise RuntimeError("--evaluate-test was requested, but no test split is available.")
        test_loader = make_loader(
            dataset.test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            torch=torch,
        )

    model, checkpoint_name = build_model(args, deps, dataset.class_names)
    model = apply_lora(model, args, deps)
    set_trainable_mode(model, args.trainable_mode)
    total_parameters, trainable_parameters = parameter_counts(model)
    if trainable_parameters == 0:
        raise RuntimeError("No trainable parameters. Check --trainable-mode and model structure.")
    model.to(device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(torch, device, args.mixed_precision)

    history: list[dict[str, Any]] = []
    training_examples = 0.0
    training_seconds = 0.0
    start = time.perf_counter()
    best_validation_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            torch=torch,
            scaler=scaler,
        )
        validation_metrics = evaluate_model(
            model=model,
            loader=validation_loader,
            device=device,
            args=args,
            torch=torch,
            class_names=dataset.class_names,
        )
        training_examples += train_metrics["examples"]
        training_seconds += train_metrics["elapsed_seconds"]
        best_validation_accuracy = max(best_validation_accuracy, validation_metrics["accuracy"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "train_examples_per_second": train_metrics["examples_per_second"],
                "validation_examples_per_second": validation_metrics["examples_per_second"],
            }
        )
        print(
            f"epoch={epoch} "
            f"train_accuracy={train_metrics['accuracy']:.4f} "
            f"validation_accuracy={validation_metrics['accuracy']:.4f}"
        )

    validation_metrics = evaluate_model(
        model=model,
        loader=validation_loader,
        device=device,
        args=args,
        torch=torch,
        class_names=dataset.class_names,
    )
    if args.epochs == 0:
        best_validation_accuracy = validation_metrics["accuracy"]

    test_metrics: dict[str, Any]
    if test_loader is not None:
        test_metrics = evaluate_model(
            model=model,
            loader=test_loader,
            device=device,
            args=args,
            torch=torch,
            class_names=dataset.class_names,
        )
    else:
        test_metrics = {"status": "not requested"}

    elapsed = time.perf_counter() - start
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation_confusion_path = args.output_dir / "validation_confusion_matrix.csv"
    validation_per_class_path = args.output_dir / "validation_per_class_accuracy.csv"
    validation_predictions_path = args.output_dir / "validation_prediction_examples.csv"
    history_path = args.output_dir / "history.csv"

    write_confusion_csv(
        validation_confusion_path,
        validation_metrics["confusion_matrix"],
        dataset.class_names,
    )
    write_dict_rows(
        validation_per_class_path,
        per_class_accuracy(validation_metrics["confusion_matrix"], dataset.class_names),
        ["class_name", "correct", "total", "accuracy"],
    )
    write_dict_rows(
        validation_predictions_path,
        validation_metrics["prediction_rows"],
        ["source", "true_label", "predicted_label", "confidence", "error_note"],
    )
    write_history_csv(history_path, history)

    artifact_paths = {
        "history": str(history_path),
        "validation_confusion_matrix": str(validation_confusion_path),
        "validation_per_class_accuracy": str(validation_per_class_path),
        "validation_prediction_examples": str(validation_predictions_path),
    }

    if args.save_mistake_gallery:
        gallery_path = args.output_dir / "validation_mistake_gallery.png"
        save_mistake_gallery(
            gallery_path,
            validation_metrics["gallery_items"],
            dataset.class_names,
        )
        if gallery_path.exists():
            artifact_paths["validation_mistake_gallery"] = str(gallery_path)

    if args.evaluate_test and "confusion_matrix" in test_metrics:
        test_confusion_path = args.output_dir / "test_confusion_matrix.csv"
        test_per_class_path = args.output_dir / "test_per_class_accuracy.csv"
        test_predictions_path = args.output_dir / "test_prediction_examples.csv"
        write_confusion_csv(test_confusion_path, test_metrics["confusion_matrix"], dataset.class_names)
        write_dict_rows(
            test_per_class_path,
            per_class_accuracy(test_metrics["confusion_matrix"], dataset.class_names),
            ["class_name", "correct", "total", "accuracy"],
        )
        write_dict_rows(
            test_predictions_path,
            test_metrics["prediction_rows"],
            ["source", "true_label", "predicted_label", "confidence", "error_note"],
        )
        artifact_paths.update(
            {
                "test_confusion_matrix": str(test_confusion_path),
                "test_per_class_accuracy": str(test_per_class_path),
                "test_prediction_examples": str(test_predictions_path),
            }
        )

    if args.save_model:
        model_path = args.output_dir / "final_model.pt"
        torch.save(model.state_dict(), model_path)
        artifact_paths["final_model"] = str(model_path)
        if args.trainable_mode == "lora" and hasattr(model, "save_pretrained"):
            adapter_path = args.output_dir / "final_lora_adapter"
            model.save_pretrained(adapter_path)
            artifact_paths["final_lora_adapter"] = str(adapter_path)

    metadata = {
        "command": command_line(argv if argv is not None else sys.argv[1:]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": args.dataset,
        "dataset_source": dataset.source,
        "class_count": len(dataset.class_names),
        "class_names": dataset.class_names,
        "split_rule": dataset.split_rule,
        "checkpoint": checkpoint_name,
        "preprocessing": {
            "image_size": preprocessing.image_size,
            "resize_size": preprocessing.resize_size,
            "mean": preprocessing.mean,
            "std": preprocessing.std,
            "summary": preprocessing.summary,
        },
        "trainable_mode": args.trainable_mode,
        "lora": lora_metadata(args),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "precision": args.mixed_precision,
        "hardware": device_summary(torch, device),
        "artifacts": artifact_paths,
    }
    metrics = {
        "validation": strip_eval_payload(validation_metrics),
        "best_validation_accuracy": best_validation_accuracy,
        "test": strip_eval_payload(test_metrics) if "accuracy" in test_metrics else test_metrics,
        "elapsed_seconds": elapsed,
        "training_examples_per_second": (
            training_examples / training_seconds if training_seconds > 0 else 0.0
        ),
        "peak_memory": peak_memory_summary(torch, device),
        "hardware": metadata["hardware"],
    }

    metadata_path = args.output_dir / "metadata.json"
    metrics_path = args.output_dir / "metrics.json"
    summary_path = args.output_dir / "result_summary.csv"
    command_path = args.output_dir / "command.txt"
    write_json(metadata_path, metadata)
    write_json(metrics_path, metrics)
    command_path.write_text(metadata["command"] + "\n", encoding="utf-8")
    write_result_summary(
        summary_path,
        args=args,
        dataset=dataset,
        preprocessing=preprocessing,
        checkpoint_name=checkpoint_name,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        metrics=metrics,
        artifact_paths=artifact_paths,
    )
    artifact_paths.update(
        {
            "metadata": str(metadata_path),
            "metrics": str(metrics_path),
            "result_summary": str(summary_path),
            "command": str(command_path),
        }
    )

    print(f"validation_accuracy={validation_metrics['accuracy']:.4f}")
    print(f"best_validation_accuracy={best_validation_accuracy:.4f}")
    if "accuracy" in test_metrics:
        print(f"test_accuracy={test_metrics['accuracy']:.4f}")
    else:
        print("test_accuracy=not_requested")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"trainable_parameters={trainable_parameters} total_parameters={total_parameters}")
    print(f"output_dir={args.output_dir}")

    return {
        "status": "ok",
        "args": vars(args),
        "metadata": metadata,
        "metrics": metrics,
        "artifacts": artifact_paths,
    }


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    args = parse_args(raw_argv)
    run_experiment(args, argv=raw_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
