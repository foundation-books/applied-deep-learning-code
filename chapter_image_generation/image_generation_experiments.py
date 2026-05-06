#!/usr/bin/env python3
"""Reference experiments for the image generation chapter.

The script is intentionally self-contained. It creates a small permitted
synthetic image dataset, trains compact image-generation models, and writes
book-facing artifacts. The goal is not to compete with large public
text-to-image systems. The goal is to produce reproducible evidence for the
chapter's workflow: reconstruction, latent sampling, adversarial generation,
diffusion denoising, and a small prompt-conditioned diffusion fine-tuning
ablation with LoRA-style adapters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - exercised in lean environments
    np = None  # type: ignore[assignment]
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None

try:
    from PIL import Image, ImageDraw, ImageFilter
except Exception as exc:  # pragma: no cover - exercised in lean environments
    Image = ImageDraw = ImageFilter = None  # type: ignore[assignment]
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None


COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("red", (218, 55, 67)),
    ("green", (58, 156, 93)),
    ("blue", (58, 107, 205)),
    ("teal", (43, 169, 167)),
)
SHAPES: tuple[str, ...] = ("circle", "square", "triangle")
STYLES: tuple[str, ...] = ("solid", "striped", "dotted")
TARGET_CONDITION = ("teal", "circle", "striped")


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    seed: int
    image_size: int
    base_samples: int
    target_samples: int
    batch_size: int
    diffusion_batch_size: int
    ae_steps: int
    vae_steps: int
    gan_steps: int
    diffusion_steps: int
    finetune_steps: int
    timesteps: int
    latent_dim: int
    lora_ranks: tuple[int, ...]
    learning_rate: float
    gan_learning_rate: float
    diffusion_learning_rate: float
    finetune_learning_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-deps", action="store_true", help="Only check optional runtime dependencies.")
    parser.add_argument(
        "--allow-missing-deps",
        action="store_true",
        help="Exit successfully from --check-deps even when optional ML dependencies are missing.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/image-generation-reference"))
    parser.add_argument("--mode", choices=["quick", "reference"], default="quick")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--base-samples", type=int, default=None)
    parser.add_argument("--target-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--diffusion-batch-size", type=int, default=None)
    parser.add_argument("--ae-steps", type=int, default=None)
    parser.add_argument("--vae-steps", type=int, default=None)
    parser.add_argument("--gan-steps", type=int, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--finetune-steps", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lora-ranks", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gan-learning-rate", type=float, default=2e-4)
    parser.add_argument("--diffusion-learning-rate", type=float, default=2e-4)
    parser.add_argument("--finetune-learning-rate", type=float, default=5e-4)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> ExperimentConfig:
    quick = args.mode == "quick"
    return ExperimentConfig(
        mode=args.mode,
        seed=args.seed,
        image_size=args.image_size,
        base_samples=args.base_samples if args.base_samples is not None else (96 if quick else 768),
        target_samples=args.target_samples if args.target_samples is not None else (18 if quick else 72),
        batch_size=args.batch_size if args.batch_size is not None else (16 if quick else 64),
        diffusion_batch_size=args.diffusion_batch_size if args.diffusion_batch_size is not None else (16 if quick else 64),
        ae_steps=args.ae_steps if args.ae_steps is not None else (4 if quick else 300),
        vae_steps=args.vae_steps if args.vae_steps is not None else (4 if quick else 400),
        gan_steps=args.gan_steps if args.gan_steps is not None else (4 if quick else 500),
        diffusion_steps=args.diffusion_steps if args.diffusion_steps is not None else (6 if quick else 1200),
        finetune_steps=args.finetune_steps if args.finetune_steps is not None else (4 if quick else 300),
        timesteps=args.timesteps if args.timesteps is not None else (20 if quick else 50),
        latent_dim=args.latent_dim,
        lora_ranks=tuple(args.lora_ranks),
        learning_rate=args.learning_rate,
        gan_learning_rate=args.gan_learning_rate,
        diffusion_learning_rate=args.diffusion_learning_rate,
        finetune_learning_rate=args.finetune_learning_rate,
    )


def check_dependencies(allow_missing: bool) -> int:
    packages = ["numpy", "PIL", "torch"]
    missing: list[str] = []
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "available")
            print(f"{package}: {version}")
        except Exception as exc:  # pragma: no cover - used for environment checks
            print(f"{package}: MISSING ({type(exc).__name__}: {exc})")
            missing.append(package)
    if missing and not allow_missing:
        return 1
    return 0


def require_core_image_dependencies() -> None:
    missing = []
    if NUMPY_IMPORT_ERROR is not None:
        missing.append(f"numpy ({type(NUMPY_IMPORT_ERROR).__name__}: {NUMPY_IMPORT_ERROR})")
    if PIL_IMPORT_ERROR is not None:
        missing.append(f"PIL ({type(PIL_IMPORT_ERROR).__name__}: {PIL_IMPORT_ERROR})")
    if missing:
        joined = "; ".join(missing)
        raise RuntimeError(f"Missing required image-generation dependencies: {joined}")


def all_conditions() -> list[tuple[str, str, str]]:
    return [(color, shape, style) for color, _ in COLORS for shape in SHAPES for style in STYLES]


def condition_to_indices(condition: tuple[str, str, str]) -> tuple[int, int, int]:
    color, shape, style = condition
    color_idx = [name for name, _ in COLORS].index(color)
    shape_idx = SHAPES.index(shape)
    style_idx = STYLES.index(style)
    return color_idx, shape_idx, style_idx


def shape_mask(size: int, condition: tuple[str, str, str], rng: random.Random) -> Image.Image:
    _, shape, _ = condition
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    margin = rng.randint(5, 8)
    jitter_x = rng.randint(-2, 2)
    jitter_y = rng.randint(-2, 2)
    box = [
        margin + jitter_x,
        margin + jitter_y,
        size - margin + jitter_x,
        size - margin + jitter_y,
    ]
    if shape == "circle":
        draw.ellipse(box, fill=255)
    elif shape == "square":
        draw.rounded_rectangle(box, radius=2, fill=255)
    else:
        x0, y0, x1, y1 = box
        points = [(size // 2 + jitter_x, y0), (x0, y1), (x1, y1)]
        draw.polygon(points, fill=255)
    return mask


def draw_synthetic_image(condition: tuple[str, str, str], seed: int, size: int) -> np.ndarray:
    color_name, _, style = condition
    rng = random.Random(seed)
    base_color = dict(COLORS)[color_name]
    bg = tuple(int(np.clip(238 + rng.randint(-8, 8), 0, 255)) for _ in range(3))
    image = Image.new("RGB", (size, size), bg)
    mask = shape_mask(size, condition, rng)

    fill = Image.new("RGB", (size, size), base_color)
    if style == "solid":
        styled = fill
    elif style == "striped":
        styled = Image.new("RGB", (size, size), tuple(min(255, int(c * 1.10)) for c in base_color))
        stripe_draw = ImageDraw.Draw(styled)
        stripe_color = tuple(max(0, int(c * 0.60)) for c in base_color)
        for offset in range(-size, size * 2, 7):
            stripe_draw.line((offset, size, offset + size, 0), fill=stripe_color, width=3)
    else:
        styled = Image.new("RGB", (size, size), tuple(min(255, int(c * 1.12)) for c in base_color))
        dot_draw = ImageDraw.Draw(styled)
        dot_color = tuple(max(0, int(c * 0.50)) for c in base_color)
        for y in range(7, size, 8):
            for x in range(7, size, 8):
                if rng.random() > 0.12:
                    dot_draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=dot_color)

    image = Image.composite(styled, image, mask)
    outline_mask = mask.filter(ImageFilter.FIND_EDGES)
    edge = Image.new("RGB", (size, size), (45, 45, 45))
    image = Image.composite(edge, image, outline_mask)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    noise = np.random.default_rng(seed + 17).normal(0.0, 0.012, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    return np.transpose(arr, (2, 0, 1))


def build_dataset(
    base_samples: int,
    target_samples: int,
    size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    base_conditions = [condition for condition in all_conditions() if condition != TARGET_CONDITION]
    base_images: list[np.ndarray] = []
    base_labels: list[tuple[int, int, int]] = []
    for i in range(base_samples):
        condition = base_conditions[i % len(base_conditions)]
        base_images.append(draw_synthetic_image(condition, seed + i * 13, size))
        base_labels.append(condition_to_indices(condition))

    target_images: list[np.ndarray] = []
    target_labels: list[tuple[int, int, int]] = []
    for i in range(target_samples):
        target_images.append(draw_synthetic_image(TARGET_CONDITION, seed + 10000 + i * 19, size))
        target_labels.append(condition_to_indices(TARGET_CONDITION))

    images = np.stack(base_images + target_images)
    labels = np.asarray(base_labels + target_labels, dtype=np.int64)
    return images, labels, base_conditions, [TARGET_CONDITION]


def save_grid(path: Path, tensors: Any, nrow: int = 8, padding: int = 2) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(tensors, torch.Tensor):
        arr = tensors.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensors, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.min() < -0.05:
        arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    count, channels, height, width = arr.shape
    nrow = max(1, min(nrow, count))
    ncol = int(math.ceil(count / nrow))
    grid = np.ones(
        (ncol * height + (ncol + 1) * padding, nrow * width + (nrow + 1) * padding, channels),
        dtype=np.float32,
    )
    for idx in range(count):
        row, col = divmod(idx, nrow)
        y = padding + row * (height + padding)
        x = padding + col * (width + padding)
        grid[y : y + height, x : x + width, :] = np.transpose(arr[idx], (1, 2, 0))
    Image.fromarray((grid * 255).astype(np.uint8)).save(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def command_string() -> str:
    command = " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
    home = str(Path.home())
    if home and home in command:
        command = command.replace(home, "<home>")
    return command


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def cuda_peak_mb(torch_module: Any, device: Any) -> float | None:
    if getattr(device, "type", None) != "cuda":
        return None
    return float(torch_module.cuda.max_memory_allocated(device) / (1024**2))


def mps_available(torch_module: Any) -> bool:
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    return bool(mps is not None and mps.is_available())


def resolve_device(torch_module: Any, args: argparse.Namespace) -> Any:
    if args.device == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available")
        return torch_module.device("cuda")
    if args.device == "mps":
        if not mps_available(torch_module):
            raise RuntimeError(
                "--device mps was requested, but PyTorch MPS is not available"
            )
        return torch_module.device("mps")
    if args.device == "cpu":
        return torch_module.device("cpu")
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if args.mode == "quick" and mps_available(torch_module):
        return torch_module.device("mps")
    return torch_module.device("cpu")


def device_display_name(torch_module: Any, device: Any) -> str:
    if device.type == "cuda":
        return torch_module.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal Performance Shaders (mps)"
    return "CPU"


def model_parameter_count(model: Any, trainable_only: bool = False) -> int:
    total = 0
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        total += int(param.numel())
    return total


def make_schedule(torch: Any, timesteps: int, device: Any) -> dict[str, Any]:
    betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": torch.sqrt(alpha_bars),
        "sqrt_one_minus_alpha_bars": torch.sqrt(1.0 - alpha_bars),
    }


def define_torch_models(torch: Any, nn: Any, F: Any, cfg: ExperimentConfig) -> dict[str, Any]:
    image_dim = 3 * cfg.image_size * cfg.image_size

    class Autoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(image_dim, 512),
                nn.ReLU(),
                nn.Linear(512, cfg.latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(cfg.latent_dim, 512),
                nn.ReLU(),
                nn.Linear(512, image_dim),
                nn.Sigmoid(),
            )

        def forward(self, x: Any) -> Any:
            z = self.encoder(x)
            return self.decoder(z).view(-1, 3, cfg.image_size, cfg.image_size)

    class VAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(image_dim, 512), nn.ReLU())
            self.mu = nn.Linear(512, cfg.latent_dim)
            self.logvar = nn.Linear(512, cfg.latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(cfg.latent_dim, 512),
                nn.ReLU(),
                nn.Linear(512, image_dim),
                nn.Sigmoid(),
            )

        def encode(self, x: Any) -> tuple[Any, Any]:
            hidden = self.backbone(x)
            return self.mu(hidden), self.logvar(hidden)

        def reparameterize(self, mu: Any, logvar: Any) -> Any:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)

        def decode(self, z: Any) -> Any:
            return self.decoder(z).view(-1, 3, cfg.image_size, cfg.image_size)

        def forward(self, x: Any) -> tuple[Any, Any, Any]:
            mu, logvar = self.encode(x)
            return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    class Generator(nn.Module):
        def __init__(self, latent_dim: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 256 * 4 * 4),
                nn.ReLU(True),
                nn.Unflatten(1, (256, 4, 4)),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, 4, 2, 1),
                nn.Tanh(),
            )

        def forward(self, z: Any) -> Any:
            return self.net(z)

    class Discriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 1),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x).view(-1)

    class LoRAConv2d(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int,
            rank: int,
            alpha: float | None = None,
        ) -> None:
            super().__init__()
            self.base = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            self.rank = rank
            self.alpha = float(alpha if alpha is not None else rank)
            self.lora_enabled = False
            self.lora_down = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
            self.lora_up = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)

        def forward(self, x: Any) -> Any:
            y = self.base(x)
            if self.lora_enabled:
                y = y + (self.alpha / self.rank) * self.lora_up(self.lora_down(x))
            return y

    class ResBlock(nn.Module):
        def __init__(self, channels: int, cond_dim: int, rank: int) -> None:
            super().__init__()
            self.conv1 = LoRAConv2d(channels, channels, 3, 1, rank)
            self.conv2 = LoRAConv2d(channels, channels, 3, 1, rank)
            self.cond = nn.Linear(cond_dim, channels)
            self.norm1 = nn.GroupNorm(8, channels)
            self.norm2 = nn.GroupNorm(8, channels)

        def forward(self, x: Any, emb: Any) -> Any:
            bias = self.cond(emb).view(emb.shape[0], -1, 1, 1)
            h = F.silu(self.norm1(self.conv1(x)) + bias)
            h = self.norm2(self.conv2(h))
            return F.silu(x + h)

    class TinyConditionedDenoiser(nn.Module):
        def __init__(self, rank: int = 4, channels: int = 64, cond_dim: int = 64) -> None:
            super().__init__()
            self.color_emb = nn.Embedding(len(COLORS), cond_dim)
            self.shape_emb = nn.Embedding(len(SHAPES), cond_dim)
            self.style_emb = nn.Embedding(len(STYLES), cond_dim)
            self.time_emb = nn.Embedding(cfg.timesteps, cond_dim)
            self.in_conv = LoRAConv2d(3, channels, 3, 1, rank)
            self.block1 = ResBlock(channels, cond_dim, rank)
            self.block2 = ResBlock(channels, cond_dim, rank)
            self.block3 = ResBlock(channels, cond_dim, rank)
            self.out = nn.Conv2d(channels, 3, 3, padding=1)

        def condition_embedding(self, cond: Any, t: Any) -> Any:
            color = self.color_emb(cond[:, 0])
            shape = self.shape_emb(cond[:, 1])
            style = self.style_emb(cond[:, 2])
            return color + shape + style + self.time_emb(t)

        def set_lora_enabled(self, enabled: bool) -> None:
            for module in self.modules():
                if isinstance(module, LoRAConv2d):
                    module.lora_enabled = enabled

        def freeze_for_lora(self) -> None:
            for param in self.parameters():
                param.requires_grad = False
            for module in self.modules():
                if isinstance(module, LoRAConv2d):
                    module.lora_enabled = True
                    module.lora_down.weight.requires_grad = True
                    module.lora_up.weight.requires_grad = True

        def forward(self, x: Any, t: Any, cond: Any) -> Any:
            emb = self.condition_embedding(cond, t)
            h = F.silu(self.in_conv(x))
            h = self.block1(h, emb)
            h = self.block2(h, emb)
            h = self.block3(h, emb)
            return self.out(F.silu(h))

    return {
        "Autoencoder": Autoencoder,
        "VAE": VAE,
        "Generator": Generator,
        "Discriminator": Discriminator,
        "TinyConditionedDenoiser": TinyConditionedDenoiser,
    }


def copy_matching_base_weights(source: dict[str, Any], target: Any) -> None:
    target_state = target.state_dict()
    copied = {}
    for name, tensor in source.items():
        if "lora_" in name:
            continue
        if name in target_state and target_state[name].shape == tensor.shape:
            copied[name] = tensor
    target_state.update(copied)
    target.load_state_dict(target_state)


def sample_batch(torch: Any, images: Any, conds: Any, batch_size: int, device: Any) -> tuple[Any, Any]:
    idx = torch.randint(0, images.shape[0], (batch_size,))
    return images[idx].to(device), conds[idx].to(device)


def train_autoencoder(torch: Any, nn: Any, F: Any, models: dict[str, Any], images: Any, cfg: ExperimentConfig, device: Any) -> tuple[Any, dict[str, Any]]:
    model = models["Autoencoder"]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    start = time.perf_counter()
    if getattr(device, "type", None) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    final_loss = 0.0
    for _ in range(cfg.ae_steps):
        idx = torch.randint(0, images.shape[0], (cfg.batch_size,))
        x = images[idx].to(device)
        recon = model(x)
        loss = F.mse_loss(recon, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu())
    seconds = time.perf_counter() - start
    with torch.no_grad():
        examples = images[:8].to(device)
        recon = model(examples).detach().cpu()
    return recon, {"component": "autoencoder", "seconds": seconds, "peak_memory_mb": cuda_peak_mb(torch, device), "final_loss": final_loss}


def train_vae(torch: Any, nn: Any, F: Any, models: dict[str, Any], images: Any, cfg: ExperimentConfig, device: Any) -> tuple[Any, dict[str, Any]]:
    model = models["VAE"]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    start = time.perf_counter()
    if getattr(device, "type", None) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    final_loss = 0.0
    for _ in range(cfg.vae_steps):
        idx = torch.randint(0, images.shape[0], (cfg.batch_size,))
        x = images[idx].to(device)
        recon, mu, logvar = model(x)
        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 1e-3 * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu())
    seconds = time.perf_counter() - start
    with torch.no_grad():
        x0 = images[0:1].to(device)
        x1 = images[min(9, images.shape[0] - 1) : min(10, images.shape[0])].to(device)
        mu0, _ = model.encode(x0)
        mu1, _ = model.encode(x1)
        weights = torch.linspace(0, 1, 8, device=device).view(-1, 1)
        z = (1 - weights) * mu0 + weights * mu1
        interp = model.decode(z).detach().cpu()
    return interp, {"component": "vae", "seconds": seconds, "peak_memory_mb": cuda_peak_mb(torch, device), "final_loss": final_loss}


def train_gan(torch: Any, nn: Any, F: Any, models: dict[str, Any], images: Any, cfg: ExperimentConfig, device: Any) -> tuple[Any, dict[str, Any]]:
    latent_dim = 64
    generator = models["Generator"](latent_dim).to(device)
    discriminator = models["Discriminator"]().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.gan_learning_rate, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.gan_learning_rate, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()
    start = time.perf_counter()
    if getattr(device, "type", None) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    final_g = 0.0
    final_d = 0.0
    scaled = images * 2.0 - 1.0
    for _ in range(cfg.gan_steps):
        idx = torch.randint(0, scaled.shape[0], (cfg.batch_size,))
        real = scaled[idx].to(device)
        z = torch.randn(cfg.batch_size, latent_dim, device=device)
        fake = generator(z).detach()
        real_logits = discriminator(real)
        fake_logits = discriminator(fake)
        d_loss = loss_fn(real_logits, torch.full_like(real_logits, 0.9)) + loss_fn(fake_logits, torch.zeros_like(fake_logits))
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

        z = torch.randn(cfg.batch_size, latent_dim, device=device)
        fake = generator(z)
        fake_logits = discriminator(fake)
        g_loss = loss_fn(fake_logits, torch.ones_like(fake_logits))
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()
        final_g = float(g_loss.detach().cpu())
        final_d = float(d_loss.detach().cpu())
    seconds = time.perf_counter() - start
    with torch.no_grad():
        samples = generator(torch.randn(16, latent_dim, device=device)).detach().cpu()
    return samples, {
        "component": "dcgan",
        "seconds": seconds,
        "peak_memory_mb": cuda_peak_mb(torch, device),
        "final_loss": final_g,
        "discriminator_loss": final_d,
    }


def add_noise(torch: Any, schedule: dict[str, Any], x0: Any, t: Any) -> tuple[Any, Any]:
    noise = torch.randn_like(x0)
    sqrt_ab = schedule["sqrt_alpha_bars"][t].view(-1, 1, 1, 1)
    sqrt_om = schedule["sqrt_one_minus_alpha_bars"][t].view(-1, 1, 1, 1)
    return sqrt_ab * x0 + sqrt_om * noise, noise


def train_diffusion_base(
    torch: Any,
    F: Any,
    models: dict[str, Any],
    images: Any,
    conds: Any,
    cfg: ExperimentConfig,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    model = models["TinyConditionedDenoiser"](rank=max(cfg.lora_ranks)).to(device)
    model.set_lora_enabled(False)
    base_params = [param for name, param in model.named_parameters() if "lora_" not in name]
    opt = torch.optim.AdamW(base_params, lr=cfg.diffusion_learning_rate, weight_decay=1e-4)
    schedule = make_schedule(torch, cfg.timesteps, device)
    start = time.perf_counter()
    if getattr(device, "type", None) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    final_loss = 0.0
    scaled = images * 2.0 - 1.0
    for _ in range(cfg.diffusion_steps):
        x0, cond = sample_batch(torch, scaled, conds, cfg.diffusion_batch_size, device)
        t = torch.randint(0, cfg.timesteps, (x0.shape[0],), device=device)
        xt, noise = add_noise(torch, schedule, x0, t)
        pred = model(xt, t, cond)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu())
    seconds = time.perf_counter() - start
    return model, {
        "component": "prompt_conditioned_diffusion_pretrain",
        "seconds": seconds,
        "peak_memory_mb": cuda_peak_mb(torch, device),
        "final_loss": final_loss,
        "trainable_parameters": sum(param.numel() for param in base_params),
    }


def sample_diffusion(
    torch: Any,
    model: Any,
    schedule: dict[str, Any],
    cond: Any,
    cfg: ExperimentConfig,
    device: Any,
    keep_trajectory: bool = False,
) -> tuple[Any, list[Any]]:
    model.eval()
    x = torch.randn(cond.shape[0], 3, cfg.image_size, cfg.image_size, device=device)
    trajectory: list[Any] = []
    with torch.no_grad():
        for step in reversed(range(cfg.timesteps)):
            t = torch.full((cond.shape[0],), step, dtype=torch.long, device=device)
            pred_noise = model(x, t, cond)
            alpha_bar_t = schedule["alpha_bars"][step]
            alpha_bar_prev = schedule["alpha_bars"][step - 1] if step > 0 else torch.tensor(1.0, device=device)
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            pred_x0 = pred_x0.clamp(-1.0, 1.0)
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
            if keep_trajectory and step in {cfg.timesteps - 1, int(cfg.timesteps * 0.75), int(cfg.timesteps * 0.5), int(cfg.timesteps * 0.25), 0}:
                trajectory.append(x.detach().cpu())
    return x.detach().cpu(), trajectory


def target_condition_tensor(torch: Any, n: int, device: Any | None = None) -> Any:
    cond = torch.tensor([condition_to_indices(TARGET_CONDITION)] * n, dtype=torch.long)
    return cond if device is None else cond.to(device)


def validation_condition_tensor(torch: Any, device: Any | None = None) -> tuple[Any, list[str]]:
    conditions = [
        TARGET_CONDITION,
        ("red", "circle", "striped"),
        ("teal", "circle", "solid"),
        ("blue", "square", "dotted"),
        ("green", "triangle", "solid"),
        TARGET_CONDITION,
        ("red", "square", "striped"),
        ("teal", "triangle", "dotted"),
    ]
    labels = [" ".join(condition) for condition in conditions]
    cond = torch.tensor([condition_to_indices(c) for c in conditions], dtype=torch.long)
    return (cond if device is None else cond.to(device)), labels


def train_lora_finetune(
    torch: Any,
    F: Any,
    models: dict[str, Any],
    base_state: dict[str, Any],
    target_images: Any,
    target_conds: Any,
    cfg: ExperimentConfig,
    device: Any,
    rank: int,
) -> tuple[Any, dict[str, Any]]:
    model = models["TinyConditionedDenoiser"](rank=rank).to(device)
    copy_matching_base_weights(base_state, model)
    model.freeze_for_lora()
    trainable = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.finetune_learning_rate, weight_decay=0.0)
    schedule = make_schedule(torch, cfg.timesteps, device)
    scaled = target_images * 2.0 - 1.0
    start = time.perf_counter()
    if getattr(device, "type", None) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    final_loss = 0.0
    for _ in range(cfg.finetune_steps):
        x0, cond = sample_batch(torch, scaled, target_conds, cfg.diffusion_batch_size, device)
        t = torch.randint(0, cfg.timesteps, (x0.shape[0],), device=device)
        xt, noise = add_noise(torch, schedule, x0, t)
        pred = model(xt, t, cond)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu())
    seconds = time.perf_counter() - start
    return model, {
        "rank": rank,
        "component": f"lora_finetune_rank_{rank}",
        "seconds": seconds,
        "peak_memory_mb": cuda_peak_mb(torch, device),
        "final_loss": final_loss,
        "trainable_parameters": model_parameter_count(model, trainable_only=True),
    }


def write_metadata(
    path: Path,
    cfg: ExperimentConfig,
    device_name: str,
    software: dict[str, str],
    output_files: list[str],
    runtime_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "Image generation chapter reference run",
        f"Run ID: {path.parent.name}",
        "Host: not recorded in public metadata",
        f"Git revision: {git_revision()}",
        f"Command: {command_string()}",
        f"Dataset: procedurally generated 32x32 RGB shapes",
        "Dataset license/provenance: generated locally by this script; no external images",
        f"Target concept: {' '.join(TARGET_CONDITION)}",
        f"Base samples: {cfg.base_samples}",
        f"Target samples: {cfg.target_samples}",
        f"Device: {device_name}",
        "Software:",
    ]
    lines.extend(f"- {name}: {version}" for name, version in software.items())
    runtime_seconds = sum(float(row.get("seconds", 0.0) or 0.0) for row in runtime_rows)
    ablation_seconds = sum(float(row.get("seconds", 0.0) or 0.0) for row in ablation_rows)
    lines.append(f"Runtime component subtotal seconds: {runtime_seconds:.4f}")
    lines.append(f"LoRA ablation subtotal seconds: {ablation_seconds:.4f}")
    lines.append(f"Full reference script seconds: {runtime_seconds + ablation_seconds:.4f}")
    lines.append("Runtime records:")
    for row in runtime_rows:
        lines.append(f"- {row}")
    lines.append("Ablation records:")
    for row in ablation_rows:
        lines.append(f"- {row}")
    lines.append("Output files:")
    lines.extend(f"- {name}" for name in output_files)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiments(args: argparse.Namespace) -> int:
    require_core_image_dependencies()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cfg = make_config(args)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = resolve_device(torch, args)
    device_name = device_display_name(torch, device)
    if device.type == "mps" and args.mode == "reference":
        print(
            "[WARN] Reference mode on MPS is allowed but not used for the "
            "published reference artifacts."
        )
    print(f"Using device: {device_name}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    images_np, labels_np, base_conditions, target_conditions = build_dataset(
        cfg.base_samples,
        cfg.target_samples,
        cfg.image_size,
        cfg.seed,
    )
    images = torch.tensor(images_np, dtype=torch.float32)
    conds = torch.tensor(labels_np, dtype=torch.long)
    target_start = cfg.base_samples
    target_images = images[target_start:]
    target_conds = conds[target_start:]
    save_grid(figures_dir / "thor1-imagegen-dataset-samples.png", images[:32], nrow=8)

    models = define_torch_models(torch, nn, F, cfg)
    runtime_rows: list[dict[str, Any]] = []
    output_files: list[str] = []

    recon, row = train_autoencoder(torch, nn, F, models, images, cfg, device)
    runtime_rows.append(row)
    save_grid(figures_dir / "thor1-imagegen-baseline-reconstructions.png", torch.cat([images[:8], recon], dim=0), nrow=8)
    output_files.append("figures/thor1-imagegen-baseline-reconstructions.png")

    interp, row = train_vae(torch, nn, F, models, images, cfg, device)
    runtime_rows.append(row)
    save_grid(figures_dir / "thor1-imagegen-vae-latent-interpolation.png", interp, nrow=8)
    output_files.append("figures/thor1-imagegen-vae-latent-interpolation.png")

    gan_samples, row = train_gan(torch, nn, F, models, images, cfg, device)
    runtime_rows.append(row)
    save_grid(figures_dir / "thor1-imagegen-dcgan-sample-grid.png", gan_samples, nrow=8)
    output_files.append("figures/thor1-imagegen-dcgan-sample-grid.png")

    schedule = make_schedule(torch, cfg.timesteps, device)
    clean = target_images[0:1].to(device) * 2.0 - 1.0
    noising_frames = []
    for t_value in np.linspace(0, cfg.timesteps - 1, 8, dtype=int):
        t = torch.full((1,), int(t_value), dtype=torch.long, device=device)
        xt, _ = add_noise(torch, schedule, clean, t)
        noising_frames.append(xt.detach().cpu())
    save_grid(figures_dir / "thor1-imagegen-diffusion-noising-grid.png", torch.cat(noising_frames, dim=0), nrow=8)
    output_files.append("figures/thor1-imagegen-diffusion-noising-grid.png")

    base_diffusion, row = train_diffusion_base(torch, F, models, images[: cfg.base_samples], conds[: cfg.base_samples], cfg, device)
    runtime_rows.append(row)
    base_state = {name: tensor.detach().cpu().clone() for name, tensor in base_diffusion.state_dict().items()}
    val_cond, prompt_labels = validation_condition_tensor(torch, device)
    base_samples, trajectory = sample_diffusion(torch, base_diffusion, schedule, val_cond, cfg, device, keep_trajectory=True)
    save_grid(figures_dir / "thor1-imagegen-pretrained-baseline-grid.png", base_samples, nrow=8)
    output_files.append("figures/thor1-imagegen-pretrained-baseline-grid.png")
    if trajectory:
        save_grid(figures_dir / "thor1-imagegen-diffusion-denoising-grid.png", torch.cat(trajectory, dim=0), nrow=len(trajectory))
        output_files.append("figures/thor1-imagegen-diffusion-denoising-grid.png")

    ablation_rows: list[dict[str, Any]] = []
    finetuned_models: list[tuple[Any, dict[str, Any], Any]] = []
    for rank in cfg.lora_ranks:
        model, row = train_lora_finetune(torch, F, models, base_state, target_images, target_conds, cfg, device, rank)
        samples, _ = sample_diffusion(torch, model, make_schedule(torch, cfg.timesteps, device), val_cond, cfg, device)
        sample_file = f"figures/thor1-imagegen-lora-rank-{rank}-grid.png"
        save_grid(output_dir / sample_file, samples, nrow=8)
        row["sample_file"] = sample_file
        ablation_rows.append(row)
        finetuned_models.append((model, row, samples))
        output_files.append(sample_file)

    selected_model, selected_row, selected_samples = min(finetuned_models, key=lambda item: float(item[1]["final_loss"]))
    save_grid(figures_dir / "thor1-imagegen-finetuned-grid.png", selected_samples, nrow=8)
    output_files.append("figures/thor1-imagegen-finetuned-grid.png")
    ablation_grid = torch.cat([item[2] for item in finetuned_models], dim=0)
    save_grid(figures_dir / "thor1-imagegen-ablation-outputs.png", ablation_grid, nrow=8)
    output_files.append("figures/thor1-imagegen-ablation-outputs.png")

    hard_cond = torch.tensor(
        [condition_to_indices(("teal", "triangle", "dotted")), condition_to_indices(("green", "circle", "striped"))] * 4,
        dtype=torch.long,
        device=device,
    )
    failure_samples, _ = sample_diffusion(torch, selected_model, make_schedule(torch, cfg.timesteps, device), hard_cond, cfg, device)
    save_grid(figures_dir / "thor1-imagegen-failure-case-grid.png", failure_samples, nrow=8)
    output_files.append("figures/thor1-imagegen-failure-case-grid.png")

    start_infer = time.perf_counter()
    _samples, _ = sample_diffusion(torch, selected_model, make_schedule(torch, cfg.timesteps, device), val_cond, cfg, device)
    inference_time = (time.perf_counter() - start_infer) / val_cond.shape[0]

    runtime_rows.append(
        {
            "component": "selected_finetuned_inference",
            "seconds": inference_time * val_cond.shape[0],
            "peak_memory_mb": cuda_peak_mb(torch, device),
            "final_loss": "",
            "seconds_per_image": inference_time,
        }
    )
    write_csv(output_dir / "thor1-imagegen-runtime-memory-table.csv", runtime_rows)
    write_csv(output_dir / "thor1-imagegen-ablation-table.csv", ablation_rows)
    output_files.extend(["thor1-imagegen-runtime-memory-table.csv", "thor1-imagegen-ablation-table.csv"])

    software = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "platform": platform.platform(),
    }
    runtime_seconds = float(sum(float(row.get("seconds", 0.0) or 0.0) for row in runtime_rows))
    ablation_seconds = float(sum(float(row.get("seconds", 0.0) or 0.0) for row in ablation_rows))
    selected_lora_seconds = float(selected_row.get("seconds", 0.0) or 0.0)
    result_values = {
        "run_id": output_dir.name,
        "host": "not-recorded",
        "device": device_name,
        "dataset_name": "procedural-shapes-32",
        "dataset_note": "Generated locally by image_generation_experiments.py; no external image data.",
        "image_count": int(images.shape[0]),
        "base_image_count": cfg.base_samples,
        "target_image_count": cfg.target_samples,
        "base_model_id": "tiny prompt-conditioned DDPM trained in this run",
        "target_concept": " ".join(TARGET_CONDITION),
        "validation_prompts": prompt_labels,
        "selected_lora_rank": int(selected_row["rank"]),
        "selected_lora_trainable_parameters": int(selected_row["trainable_parameters"]),
        "training_command": command_string(),
        "training_time_seconds": runtime_seconds + ablation_seconds,
        "runtime_component_seconds": runtime_seconds,
        "lora_ablation_seconds": ablation_seconds,
        "selected_rank_end_to_end_seconds": runtime_seconds + selected_lora_seconds,
        "training_time_note": (
            "training_time_seconds includes runtime-memory components plus all LoRA "
            "rank-ablation fine-tunes; runtime_component_seconds is the pre-ablation subtotal."
        ),
        "peak_memory_mb": max(
            float(row.get("peak_memory_mb") or 0.0) for row in [*runtime_rows, *ablation_rows]
        ),
        "inference_time_seconds_per_image": float(inference_time),
        "software_stack": software,
        "config": asdict(cfg),
    }
    (output_dir / "thor1-imagegen-result-values.json").write_text(json.dumps(result_values, indent=2), encoding="utf-8")
    output_files.append("thor1-imagegen-result-values.json")
    write_metadata(
        output_dir / "thor1-imagegen-reference-results-metadata.txt",
        cfg,
        device_name,
        software,
        output_files,
        runtime_rows,
        ablation_rows,
    )
    output_files.append("thor1-imagegen-reference-results-metadata.txt")

    print(json.dumps(result_values, indent=2))
    print(f"Wrote artifacts to {output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    if args.check_deps:
        return check_dependencies(args.allow_missing_deps)
    return run_experiments(args)


if __name__ == "__main__":
    raise SystemExit(main())
