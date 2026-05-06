"""Prepare a small RSNA pneumonia YOLO-format subset.

The script expects the RSNA `stage_2_train_labels.csv` file and downloads only
the selected DICOM images from a Hugging Face mirror. It writes a YOLO dataset
with one class, `opacity`, and keeps negative images as empty label files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


HF_DATASET_ID = "Baldezo313/rsna-pneumonia-dataset"
HF_REVISION = "b18d718027db812b3c4a88e5bfe166bca3d1ee42"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/{HF_REVISION}"
IMAGE_FOLDERS = ("stage_2_train_images_0", "stage_2_train_images_1", "stage_2_train_images_2")
DOWNLOAD_TIMEOUT_SECONDS = 60.0
USER_AGENT = "adl-textbook-companion-code/0.1"


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float


@dataclass
class Example:
    patient_id: str
    boxes: list[Box]


def load_examples(labels_csv: Path) -> dict[str, Example]:
    examples: dict[str, Example] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_id = row["patientId"]
            example = examples.setdefault(patient_id, Example(patient_id=patient_id, boxes=[]))
            if row["Target"] == "1":
                example.boxes.append(
                    Box(
                        x=float(row["x"]),
                        y=float(row["y"]),
                        width=float(row["width"]),
                        height=float(row["height"]),
                    )
                )
    return examples


def split_examples(
    examples: dict[str, Example],
    positive_counts: dict[str, int],
    negative_counts: dict[str, int],
    seed: int,
) -> dict[str, list[Example]]:
    positives = [example for example in examples.values() if example.boxes]
    negatives = [example for example in examples.values() if not example.boxes]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    splits: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    positive_offset = 0
    negative_offset = 0
    for split in ("train", "val", "test"):
        pos_count = positive_counts[split]
        neg_count = negative_counts[split]
        split_positives = positives[positive_offset : positive_offset + pos_count]
        split_negatives = negatives[negative_offset : negative_offset + neg_count]
        positive_offset += pos_count
        negative_offset += neg_count
        if len(split_positives) != pos_count or len(split_negatives) != neg_count:
            raise SystemExit("Requested subset is larger than the available RSNA label table.")
        combined = split_positives + split_negatives
        rng.shuffle(combined)
        splits[split] = combined
    return splits


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(url: str, output_path: Path, timeout: float = DOWNLOAD_TIMEOUT_SECONDS) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def download_dicom(patient_id: str, cache_dir: Path, min_bytes: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{patient_id}.dcm"
    if cached.exists():
        if cached.stat().st_size >= min_bytes:
            return cached
        cached.unlink()
    encoded = quote(f"{patient_id}.dcm")
    attempts: list[str] = []
    for folder in IMAGE_FOLDERS:
        url = f"{HF_BASE_URL}/{folder}/{encoded}"
        tmp_path = cached.with_suffix(".dcm.tmp")
        try:
            download_url(url, tmp_path)
            if tmp_path.exists() and tmp_path.stat().st_size >= min_bytes:
                tmp_path.replace(cached)
                return cached
            size = tmp_path.stat().st_size if tmp_path.exists() else 0
            attempts.append(f"{url}: downloaded {size} bytes, expected at least {min_bytes}")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            attempts.append(f"{url}: {exc}")
        finally:
            tmp_path.unlink(missing_ok=True)
    details = "; ".join(attempts) if attempts else "no download attempts were made"
    raise FileNotFoundError(
        f"Could not download DICOM for {patient_id}. Check network access, or manually place "
        f"{patient_id}.dcm in --cache-dir ({cache_dir}). Attempts: {details}"
    )


def dicom_to_png(dicom_path: Path, image_path: Path) -> tuple[int, int]:
    import numpy as np
    from PIL import Image
    import pydicom

    dataset = pydicom.dcmread(str(dicom_path))
    pixels = dataset.pixel_array.astype(np.float32)
    pixels -= float(pixels.min())
    max_value = float(pixels.max())
    if max_value > 0:
        pixels /= max_value
    pixels = (pixels * 255.0).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(pixels).convert("RGB")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    return image.width, image.height


def yolo_lines(example: Example, width: int, height: int) -> list[str]:
    lines = []
    for box in example.boxes:
        x_center = (box.x + box.width / 2.0) / width
        y_center = (box.y + box.height / 2.0) / height
        norm_width = box.width / width
        norm_height = box.height / height
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
    return lines


def write_dataset_yaml(output_dir: Path) -> None:
    yaml_path = output_dir / "rsna-pneumonia-yolo.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: opacity",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_manifest(
    output_dir: Path,
    splits: dict[str, list[Example]],
    payload: dict[str, Any],
    dicom_metadata: dict[str, dict[str, Any]],
) -> None:
    rows = []
    for split, examples in splits.items():
        for example in examples:
            metadata = dicom_metadata.get(example.patient_id, {})
            rows.append(
                {
                    "split": split,
                    "patient_id": example.patient_id,
                    "target": int(bool(example.boxes)),
                    "box_count": len(example.boxes),
                    "dicom_bytes": metadata.get("bytes", ""),
                    "dicom_sha256": metadata.get("sha256", ""),
                }
            )
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "patient_id", "target", "box_count", "dicom_bytes", "dicom_sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> int:
    labels_csv = Path(args.labels_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    positive_counts = {"train": args.train_pos, "val": args.val_pos, "test": args.test_pos}
    negative_counts = {"train": args.train_neg, "val": args.val_neg, "test": args.test_neg}
    examples = load_examples(labels_csv)
    splits = split_examples(examples, positive_counts, negative_counts, args.seed)
    dicom_metadata: dict[str, dict[str, Any]] = {}

    for split, split_examples_ in splits.items():
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index, example in enumerate(split_examples_, start=1):
            dicom_path = download_dicom(example.patient_id, cache_dir, args.min_dicom_bytes)
            dicom_metadata[example.patient_id] = {
                "bytes": dicom_path.stat().st_size,
                "sha256": sha256_file(dicom_path),
            }
            image_path = output_dir / "images" / split / f"{example.patient_id}.png"
            label_path = output_dir / "labels" / split / f"{example.patient_id}.txt"
            width, height = dicom_to_png(dicom_path, image_path)
            label_path.write_text("\n".join(yolo_lines(example, width, height)) + ("\n" if example.boxes else ""), encoding="utf-8")
            if index % args.progress_every == 0:
                print(f"{split}: prepared {index}/{len(split_examples_)} images", flush=True)

    write_dataset_yaml(output_dir)
    write_manifest(
        output_dir,
        splits,
        {
            "source": f"{HF_DATASET_ID} Hugging Face mirror of RSNA Pneumonia Detection Challenge files",
            "source_url": f"https://huggingface.co/datasets/{HF_DATASET_ID}",
            "source_revision": HF_REVISION,
            "labels_csv": labels_csv.name,
            "labels_csv_path": "not-recorded",
            "min_dicom_bytes": args.min_dicom_bytes,
            "seed": args.seed,
            "positive_counts": positive_counts,
            "negative_counts": negative_counts,
            "class_names": {"0": "opacity"},
            "public_release_note": (
                "Generated YOLO images, labels, manifests, patient identifiers, "
                "and DICOM hashes remain governed by the RSNA/NIH attribution "
                "and non-identification terms. Review those terms before "
                "publishing generated datasets or manifests."
            ),
        },
        dicom_metadata,
    )
    print(f"Wrote YOLO dataset to {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-pos", type=int, default=256)
    parser.add_argument("--train-neg", type=int, default=256)
    parser.add_argument("--val-pos", type=int, default=64)
    parser.add_argument("--val-neg", type=int, default=64)
    parser.add_argument("--test-pos", type=int, default=64)
    parser.add_argument("--test-neg", type=int, default=64)
    parser.add_argument(
        "--min-dicom-bytes",
        type=int,
        default=1024,
        help="Reject downloaded DICOM files smaller than this many bytes.",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    return prepare(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
