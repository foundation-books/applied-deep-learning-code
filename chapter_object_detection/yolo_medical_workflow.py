"""Starter workflow for the object-detection chapter homework.

The script is intentionally conservative.  It can inspect YOLO-format labels
and produce dry-run Ultralytics commands without requiring the Ultralytics
package.  Real training and validation require a prepared dataset and an
environment with the `yolo` command installed.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


BCCD_CLASS_NAMES = ["RBC", "WBC", "Platelets"]


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


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return load_simple_yolo_yaml(path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a YAML mapping.")
    return loaded


def load_simple_yolo_yaml(path: Path) -> dict[str, Any]:
    """Parse the small YAML subset used by the dataset template.

    This fallback keeps smoke checks independent of PyYAML.  Real classroom
    datasets can use PyYAML through Ultralytics, and the fallback intentionally
    rejects complex YAML rather than guessing.
    """
    config: dict[str, Any] = {}
    in_names = False
    names: dict[int, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line.startswith("  ") and in_names:
            key, separator, value = line.strip().partition(":")
            if not separator:
                raise SystemExit(f"Unsupported names entry in {path}: {raw_line}")
            names[int(key)] = value.strip().strip("'\"")
            continue
        in_names = False
        key, separator, value = line.partition(":")
        if not separator:
            raise SystemExit(f"Unsupported YAML line in {path}: {raw_line}")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key == "names" and not value:
            in_names = True
            config["names"] = names
        else:
            config[key] = value
    if names:
        config["names"] = names
    return config


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dataset_root(config: dict[str, Any], yaml_path: Path) -> Path:
    root_value = config.get("path", yaml_path.parent)
    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root


def names_from_config(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names", {})
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    raise SystemExit("The data YAML must define `names` as a list or mapping.")


def split_path(config: dict[str, Any], yaml_path: Path, split: str) -> Path:
    if split not in config:
        raise SystemExit(f"The data YAML does not define a `{split}` split.")
    value = Path(str(config[split])).expanduser()
    if value.is_absolute():
        return value
    return dataset_root(config, yaml_path) / value


def label_dir_for_split(config: dict[str, Any], yaml_path: Path, split: str) -> Path:
    image_path = split_path(config, yaml_path, split)
    parts = list(image_path.parts)
    if "images" in parts:
        index = parts.index("images")
        return Path(*parts[:index], "labels", *parts[index + 1 :])
    if image_path.name == split and image_path.parent.name == "images":
        return image_path.parent.parent / "labels" / split
    return dataset_root(config, yaml_path) / "labels" / split


def parse_label_line(
    line: str, class_count: int
) -> tuple[int, float, float, float, float]:
    pieces = line.split()
    if len(pieces) != 5:
        raise ValueError("expected five fields: class x_center y_center width height")
    class_id = int(pieces[0])
    values = tuple(float(piece) for piece in pieces[1:])
    if class_id < 0 or class_id >= class_count:
        raise ValueError(f"class id {class_id} is outside 0..{class_count - 1}")
    x_center, y_center, width, height = values
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        raise ValueError("box center coordinates must be normalized into [0, 1]")
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        raise ValueError("box width and height must be normalized and positive")
    return class_id, x_center, y_center, width, height


def inspect_labels(args: argparse.Namespace) -> int:
    yaml_path = Path(args.data_yaml).expanduser().resolve()
    config = load_yaml(yaml_path)
    names = names_from_config(config)
    label_dir = label_dir_for_split(config, yaml_path, args.split)
    label_files = sorted(label_dir.glob("*.txt"))
    if args.max_files:
        label_files = label_files[: args.max_files]

    class_counts: Counter[int] = Counter()
    invalid: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    empty_files = 0
    objects = 0

    for label_file in label_files:
        lines = [
            line.strip()
            for line in label_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            empty_files += 1
        for line_number, line in enumerate(lines, start=1):
            try:
                class_id, x_center, y_center, width, height = parse_label_line(
                    line, len(names)
                )
            except (TypeError, ValueError) as exc:
                invalid.append(
                    {
                        "file": str(label_file),
                        "line": line_number,
                        "label": line,
                        "error": str(exc),
                    }
                )
                continue
            class_counts[class_id] += 1
            objects += 1
            if len(examples) < args.example_rows:
                examples.append(
                    {
                        "file": str(label_file),
                        "class_id": class_id,
                        "class_name": names[class_id],
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                    }
                )

    summary = {
        "data_yaml": str(yaml_path),
        "split": args.split,
        "label_dir": str(label_dir),
        "files_checked": len(label_files),
        "empty_label_files": empty_files,
        "objects": objects,
        "class_counts": {names[key]: class_counts.get(key, 0) for key in sorted(names)},
        "invalid_label_count": len(invalid),
        "invalid_examples": invalid[: args.example_rows],
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    write_json(output_dir / "label_inspection.json", summary)
    with (output_dir / "label_examples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file",
                "class_id",
                "class_name",
                "x_center",
                "y_center",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerows(examples)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if invalid and not args.allow_invalid:
        return 1
    return 0


def make_template(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "medical-detection.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "# Replace the path with the prepared dataset root.",
                "path: /absolute/path/to/medical-detection",
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
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Medical Detection Dataset Template",
                "",
                "Expected YOLO layout:",
                "",
                "```text",
                "medical-detection/",
                "  images/train/*.jpg",
                "  images/val/*.jpg",
                "  images/test/*.jpg",
                "  labels/train/*.txt",
                "  labels/val/*.txt",
                "  labels/test/*.txt",
                "```",
                "",
                "Each label row is `class_id x_center y_center width height`,",
                "with normalized coordinates in `[0, 1]`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote template files to {output_dir}")
    return 0


def find_bccd_dir(source_dir: Path) -> Path:
    candidates = [source_dir, source_dir / "BCCD"]
    for candidate in candidates:
        if (candidate / "Annotations").is_dir() and (candidate / "JPEGImages").is_dir():
            return candidate
    raise SystemExit(
        "Could not find BCCD/Annotations and BCCD/JPEGImages. "
        "Pass either the cloned BCCD_Dataset repository root or its BCCD subdirectory."
    )


def read_split_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip().split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def deterministic_split(
    ids: list[str], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, list[str]]:
    if not 0.0 < train_ratio < 1.0:
        raise SystemExit("--train-ratio must be between 0 and 1.")
    if not 0.0 <= val_ratio < 1.0:
        raise SystemExit("--val-ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise SystemExit("--train-ratio plus --val-ratio must be less than 1.")
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if total == 0:
        return {"train": [], "val": [], "test": []}
    train_count = max(1, int(round(total * train_ratio)))
    val_count = int(round(total * val_ratio))
    if total >= 3:
        val_count = max(1, val_count)
    if train_count + val_count >= total and total >= 2:
        train_count = total - val_count - 1
    if train_count < 1:
        train_count = 1
    test_start = train_count + val_count
    return {
        "train": sorted(shuffled[:train_count]),
        "val": sorted(shuffled[train_count:test_start]),
        "test": sorted(shuffled[test_start:]),
    }


def bccd_splits(
    bccd_dir: Path, annotation_ids: list[str], args: argparse.Namespace
) -> dict[str, list[str]]:
    image_sets = bccd_dir / "ImageSets" / "Main"
    split_map = {
        split: read_split_file(image_sets / f"{split}.txt")
        for split in ("train", "val", "test")
    }
    if any(split_map.values()):
        allowed = set(annotation_ids)
        return {
            split: sorted(image_id for image_id in ids if image_id in allowed)
            for split, ids in split_map.items()
        }
    return deterministic_split(
        annotation_ids, args.train_ratio, args.val_ratio, args.seed
    )


def find_bccd_image(jpeg_dir: Path, image_id: str, filename: str | None) -> Path:
    candidates: list[Path] = []
    if filename:
        candidates.append(jpeg_dir / filename)
    candidates.extend(
        jpeg_dir / f"{image_id}{suffix}" for suffix in (".jpg", ".jpeg", ".png")
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"Could not find an image file for BCCD annotation {image_id}.")


def parse_bccd_annotation(
    annotation_path: Path, class_to_id: dict[str, int]
) -> tuple[str | None, list[str]]:
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    filename_node = root.find("filename")
    filename = (
        filename_node.text.strip()
        if filename_node is not None and filename_node.text
        else None
    )
    size = root.find("size")
    if size is None:
        raise SystemExit(f"{annotation_path} does not contain a VOC size element.")
    width_node = size.find("width")
    height_node = size.find("height")
    if (
        width_node is None
        or height_node is None
        or not width_node.text
        or not height_node.text
    ):
        raise SystemExit(f"{annotation_path} does not contain image width and height.")
    image_width = float(width_node.text)
    image_height = float(height_node.text)
    if image_width <= 0 or image_height <= 0:
        raise SystemExit(f"{annotation_path} has nonpositive image dimensions.")

    label_rows: list[str] = []
    for object_node in root.findall("object"):
        name_node = object_node.find("name")
        if name_node is None or not name_node.text:
            raise SystemExit(
                f"{annotation_path} contains an object without a class name."
            )
        class_name = name_node.text.strip()
        if class_name not in class_to_id:
            raise SystemExit(
                f"{annotation_path} contains unsupported BCCD class {class_name!r}."
            )
        box = object_node.find("bndbox")
        if box is None:
            raise SystemExit(
                f"{annotation_path} contains an object without a bounding box."
            )
        try:
            xmin = float(box.findtext("xmin", ""))
            ymin = float(box.findtext("ymin", ""))
            xmax = float(box.findtext("xmax", ""))
            ymax = float(box.findtext("ymax", ""))
        except ValueError as exc:
            raise SystemExit(
                f"{annotation_path} contains a nonnumeric bounding box."
            ) from exc
        xmin = min(max(xmin, 0.0), image_width)
        xmax = min(max(xmax, 0.0), image_width)
        ymin = min(max(ymin, 0.0), image_height)
        ymax = min(max(ymax, 0.0), image_height)
        if xmax <= xmin or ymax <= ymin:
            raise SystemExit(f"{annotation_path} contains a nonpositive bounding box.")
        x_center = (xmin + xmax) / (2.0 * image_width)
        y_center = (ymin + ymax) / (2.0 * image_height)
        box_width = (xmax - xmin) / image_width
        box_height = (ymax - ymin) / image_height
        label_rows.append(
            f"{class_to_id[class_name]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )
    return filename, label_rows


def prepare_bccd(args: argparse.Namespace) -> int:
    source_dir = Path(args.source_dir).expanduser().resolve()
    bccd_dir = find_bccd_dir(source_dir)
    annotation_dir = bccd_dir / "Annotations"
    jpeg_dir = bccd_dir / "JPEGImages"
    annotation_paths = sorted(annotation_dir.glob("*.xml"))
    if args.max_images:
        shuffled = list(annotation_paths)
        random.Random(args.seed).shuffle(shuffled)
        annotation_paths = sorted(shuffled[: args.max_images])
    annotation_ids = [path.stem for path in annotation_paths]
    splits = bccd_splits(bccd_dir, annotation_ids, args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    class_to_id = {name: index for index, name in enumerate(BCCD_CLASS_NAMES)}
    converted_counts: dict[str, int] = {}
    object_counts: Counter[str] = Counter()

    for split, image_ids in splits.items():
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        converted_counts[split] = 0
        for image_id in image_ids:
            annotation_path = annotation_dir / f"{image_id}.xml"
            filename, label_rows = parse_bccd_annotation(annotation_path, class_to_id)
            image_path = find_bccd_image(jpeg_dir, image_id, filename)
            target_image = output_dir / "images" / split / image_path.name
            shutil.copy2(image_path, target_image)
            (output_dir / "labels" / split / f"{image_path.stem}.txt").write_text(
                "\n".join(label_rows) + ("\n" if label_rows else ""),
                encoding="utf-8",
            )
            converted_counts[split] += 1
            for row in label_rows:
                class_id = int(row.split()[0])
                object_counts[BCCD_CLASS_NAMES[class_id]] += 1

    yaml_path = output_dir / "bccd-medical-detection.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: RBC",
                "  1: WBC",
                "  2: Platelets",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "source_dir": str(source_dir),
        "bccd_dir": str(bccd_dir),
        "dataset_yaml": str(yaml_path),
        "classes": BCCD_CLASS_NAMES,
        "image_counts": converted_counts,
        "object_counts": dict(sorted(object_counts.items())),
        "license_note": "BCCD_Dataset is distributed under the MIT license by its repository.",
        "source_url": "https://github.com/Shenggan/BCCD_Dataset",
    }
    write_json(output_dir / "bccd_conversion_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_yolo_command(args: argparse.Namespace) -> list[str]:
    if args.mode == "export":
        return ["yolo", "export", f"model={args.model}", f"format={args.export_format}"]
    if args.mode == "predict":
        return [
            "yolo",
            "detect",
            "predict",
            f"model={args.model}",
            f"source={args.source}",
            f"imgsz={args.imgsz}",
            f"project={args.project}",
            f"name={args.name}",
        ]
    mode = "val" if args.mode == "test" else args.mode
    command = [
        "yolo",
        "detect",
        mode,
        f"model={args.model}",
        f"data={args.data_yaml}",
        f"imgsz={args.imgsz}",
        f"project={args.project}",
        f"name={args.name}",
    ]
    if args.mode == "train":
        command.extend(
            [f"epochs={args.epochs}", f"batch={args.batch}", f"seed={args.seed}"]
        )
    if args.mode in {"val", "test"}:
        command.append(f"split={args.mode}")
    return command


def count_source_images(source: str) -> int | None:
    path = Path(source).expanduser()
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    if path.is_file() and path.suffix.lower() in suffixes:
        return 1
    if path.is_dir():
        return sum(
            1
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in suffixes
        )
    return None


def run_dir_from_args(args: argparse.Namespace) -> Path | None:
    if getattr(args, "project", None) and getattr(args, "name", None):
        return Path(str(args.project)).expanduser().resolve() / str(args.name)
    return None


def yolo_command(args: argparse.Namespace) -> int:
    command = build_yolo_command(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    write_json(
        output_dir / f"{args.mode}_command.json",
        {"command": command, "dry_run": args.dry_run},
    )
    print(" ".join(command))
    if args.dry_run:
        return 0
    if shutil.which(command[0]) is None:
        message = "The `yolo` command was not found. Install Ultralytics before running training."
        if args.allow_missing_deps:
            print(f"[SKIP] {message}")
            return 0
        raise SystemExit(message)
    started_at = iso_now()
    start = time.perf_counter()
    completed = subprocess.run(command, check=False)
    duration_seconds = time.perf_counter() - start
    finished_at = iso_now()
    timing: dict[str, Any] = {
        "mode": args.mode,
        "command": command,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(duration_seconds, 3),
        "return_code": completed.returncode,
    }
    if args.mode == "predict":
        image_count = count_source_images(args.source)
        timing["source_image_count"] = image_count
        if image_count:
            timing["latency_ms_per_image"] = round(
                duration_seconds * 1000.0 / image_count, 3
            )
    write_json(output_dir / f"{args.mode}_timing.json", timing)
    run_dir = run_dir_from_args(args)
    if run_dir is not None:
        write_json(run_dir / f"{args.mode}_timing.json", timing)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return 0


def value_from_row(row: dict[str, str], candidates: tuple[str, ...]) -> str:
    normalized = {key.lower().replace(" ", ""): key for key in row}
    for candidate in candidates:
        key = normalized.get(candidate.lower().replace(" ", ""))
        if key is not None:
            return row.get(key, "")
    for key, value in row.items():
        compact = key.lower().replace(" ", "")
        if any(
            candidate.lower().replace(" ", "") in compact for candidate in candidates
        ):
            return value
    return ""


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def summarize_runs(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    rows: list[dict[str, str]] = []
    for results_csv in sorted(runs_dir.rglob("results.csv")):
        with results_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            records = list(reader)
        if not records:
            continue
        final = records[-1]
        train_timing = load_json_if_exists(results_csv.parent / "train_timing.json")
        predict_timing = load_json_if_exists(results_csv.parent / "predict_timing.json")
        rows.append(
            {
                "run": str(results_csv.parent.relative_to(runs_dir)),
                "epoch": value_from_row(final, ("epoch",)),
                "precision": value_from_row(
                    final, ("metrics/precision(B)", "precision")
                ),
                "recall": value_from_row(final, ("metrics/recall(B)", "recall")),
                "map50": value_from_row(final, ("metrics/mAP50(B)", "map50")),
                "map50_95": value_from_row(
                    final, ("metrics/mAP50-95(B)", "map50-95", "map")
                ),
                "training_seconds": str(train_timing.get("duration_seconds", "")),
                "latency_ms_per_image": str(
                    predict_timing.get("latency_ms_per_image", "")
                ),
            }
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "run_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "epoch",
                "precision",
                "recall",
                "map50",
                "map50_95",
                "training_seconds",
                "latency_ms_per_image",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output_dir / "run_summary.json", {"runs_dir": str(runs_dir), "runs": rows}
    )
    print(f"Wrote {len(rows)} summarized run(s) to {summary_csv}")
    return 0


def metric_rows_from_results(runs_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for results_csv in sorted(runs_dir.rglob("results.csv")):
        with results_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for record in reader:
                rows.append(
                    {
                        "run": str(results_csv.parent.relative_to(runs_dir)),
                        "epoch": value_from_row(record, ("epoch",)),
                        "precision": value_from_row(
                            record, ("metrics/precision(B)", "precision")
                        ),
                        "recall": value_from_row(
                            record, ("metrics/recall(B)", "recall")
                        ),
                        "map50": value_from_row(record, ("metrics/mAP50(B)", "map50")),
                        "map50_95": value_from_row(
                            record, ("metrics/mAP50-95(B)", "map50-95", "map")
                        ),
                    }
                )
    return rows


def plot_metrics(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = metric_rows_from_results(runs_dir)
    metrics_csv = output_dir / "metrics_long.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run", "epoch", "precision", "recall", "map50", "map50_95"],
        )
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib  # type: ignore[import-not-found]

        matplotlib.use("Agg")
        import pandas as pd  # type: ignore[import-not-found]
        import plotnine as p9  # type: ignore[import-not-found]
        from plotnine import (  # type: ignore[import-not-found]
            aes,
            geom_line,
            geom_point,
            ggplot,
            labs,
            scale_color_manual,
        )
    except ImportError as exc:
        if args.allow_missing_deps:
            print(
                f"[SKIP] plotnine is not installed; wrote {metrics_csv} without a plot."
            )
            return 0
        raise SystemExit(
            "plotnine is required for plot-metrics. Install it or pass --allow-missing-deps."
        ) from exc

    if not rows:
        if args.allow_empty:
            print(
                f"No results.csv rows found under {runs_dir}; wrote empty {metrics_csv}."
            )
            return 0
        raise SystemExit(f"No results.csv rows found under {runs_dir}.")

    plot_path = output_dir / "metrics_map50.png"
    plot_rows = []
    for row in rows:
        epoch = parse_float(row["epoch"])
        map50 = parse_float(row["map50"])
        if epoch is not None and map50 is not None:
            plot_rows.append({"run": row["run"], "epoch": epoch, "map50": map50})
    if not plot_rows:
        if args.allow_empty:
            print(
                f"No plottable mAP50 values found under {runs_dir}; wrote {metrics_csv}."
            )
            return 0
        raise SystemExit(f"No plottable mAP50 values found under {runs_dir}.")
    plot = (
        ggplot(
            pd.DataFrame.from_records(plot_rows),
            aes("epoch", "map50", color="run", group="run"),
        )
        + geom_line(size=0.8)
        + geom_point(size=2.0)
        + scale_color_manual(
            values=PLOT_STYLE.palette_for(row["run"] for row in plot_rows)
        )
        + labs(
            title="Object detection validation mAP50", x="Epoch", y="mAP50", color="run"
        )
        + PLOT_STYLE.plot_theme(p9)
    )
    plot.save(
        plot_path,
        width=7.0,
        height=4.2,
        units="in",
        dpi=PLOT_STYLE.PLOT_DPI,
        verbose=False,
    )
    print(f"Wrote {metrics_csv} and {plot_path}")
    return 0


def environment_report(args: argparse.Namespace) -> int:
    packages: dict[str, str | None] = {}
    for package in args.packages:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    write_json(output_dir / "environment.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def smoke(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_root = output_dir / "smoke_dataset"
    for split in ("train", "val", "test"):
        (data_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (data_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (data_root / "labels/train/image_0001.txt").write_text(
        "0 0.50 0.50 0.25 0.20\n", encoding="utf-8"
    )
    (data_root / "labels/val/image_0002.txt").write_text(
        "0 0.40 0.45 0.15 0.12\n", encoding="utf-8"
    )
    (data_root / "labels/test/image_0003.txt").write_text(
        "0 0.55 0.48 0.18 0.16\n", encoding="utf-8"
    )
    data_yaml = output_dir / "smoke-medical-detection.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {data_root}",
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
    inspect_args = argparse.Namespace(
        data_yaml=str(data_yaml),
        split="train",
        output_dir=str(output_dir / "inspection"),
        max_files=20,
        example_rows=10,
        allow_invalid=False,
    )
    inspection_status = inspect_labels(inspect_args)
    bccd_root = output_dir / "synthetic_bccd" / "BCCD"
    (bccd_root / "Annotations").mkdir(parents=True, exist_ok=True)
    (bccd_root / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (bccd_root / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)
    synthetic_specs = {
        "BloodImage_00001": ("RBC", 10, 12, 34, 36, "train"),
        "BloodImage_00002": ("WBC", 20, 8, 55, 45, "val"),
        "BloodImage_00003": ("Platelets", 42, 30, 64, 52, "test"),
    }
    for image_id, (
        class_name,
        xmin,
        ymin,
        xmax,
        ymax,
        split,
    ) in synthetic_specs.items():
        (bccd_root / "JPEGImages" / f"{image_id}.jpg").write_bytes(
            b"synthetic smoke image\n"
        )
        (bccd_root / "Annotations" / f"{image_id}.xml").write_text(
            "\n".join(
                [
                    "<annotation>",
                    f"  <filename>{image_id}.jpg</filename>",
                    "  <size><width>100</width><height>80</height><depth>3</depth></size>",
                    "  <object>",
                    f"    <name>{class_name}</name>",
                    "    <bndbox>",
                    f"      <xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax>",
                    "    </bndbox>",
                    "  </object>",
                    "</annotation>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (bccd_root / "ImageSets" / "Main" / f"{split}.txt").write_text(
            f"{image_id}\n", encoding="utf-8"
        )
    bccd_output_dir = output_dir / "bccd_yolo"
    bccd_status = prepare_bccd(
        argparse.Namespace(
            source_dir=str(output_dir / "synthetic_bccd"),
            output_dir=str(bccd_output_dir),
            train_ratio=0.7,
            val_ratio=0.15,
            seed=7,
            max_images=0,
        )
    )
    bccd_inspection_status = inspect_labels(
        argparse.Namespace(
            data_yaml=str(bccd_output_dir / "bccd-medical-detection.yaml"),
            split="train",
            output_dir=str(output_dir / "bccd_inspection"),
            max_files=20,
            example_rows=10,
            allow_invalid=False,
        )
    )
    dry_run_args = argparse.Namespace(
        mode="train",
        model="yolo26n.pt",
        data_yaml=str(data_yaml),
        imgsz=64,
        epochs=1,
        batch=2,
        seed=7,
        project=str(output_dir / "runs"),
        name="smoke-dry-run",
        export_format="onnx",
        output_dir=str(output_dir / "commands"),
        dry_run=True,
        allow_missing_deps=True,
    )
    command_status = yolo_command(dry_run_args)
    smoke_run = output_dir / "runs/smoke"
    smoke_run.mkdir(parents=True, exist_ok=True)
    with (smoke_run / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": "0",
                "metrics/precision(B)": "0.10",
                "metrics/recall(B)": "0.20",
                "metrics/mAP50(B)": "0.15",
                "metrics/mAP50-95(B)": "0.08",
            }
        )
        writer.writerow(
            {
                "epoch": "1",
                "metrics/precision(B)": "0.20",
                "metrics/recall(B)": "0.30",
                "metrics/mAP50(B)": "0.25",
                "metrics/mAP50-95(B)": "0.12",
            }
        )
    write_json(smoke_run / "train_timing.json", {"duration_seconds": 1.234})
    write_json(smoke_run / "predict_timing.json", {"latency_ms_per_image": 12.3})
    summarize_status = summarize_runs(
        argparse.Namespace(
            runs_dir=str(output_dir / "runs"), output_dir=str(output_dir / "summary")
        )
    )
    plot_status = plot_metrics(
        argparse.Namespace(
            runs_dir=str(output_dir / "runs"),
            output_dir=str(output_dir / "plots"),
            allow_missing_deps=True,
            allow_empty=False,
        )
    )
    write_json(
        output_dir / "smoke_summary.json",
        {
            "inspection_status": inspection_status,
            "bccd_status": bccd_status,
            "bccd_inspection_status": bccd_inspection_status,
            "command_status": command_status,
            "summarize_status": summarize_status,
            "plot_status": plot_status,
            "data_yaml": str(data_yaml),
        },
    )
    statuses = (
        inspection_status,
        bccd_status,
        bccd_inspection_status,
        command_status,
        summarize_status,
        plot_status,
    )
    return 0 if all(status == 0 for status in statuses) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser(
        "make-template", help="Write a YOLO dataset YAML template."
    )
    template_parser.add_argument("--output-dir", default="object-detection-template")
    template_parser.set_defaults(func=make_template)

    bccd_parser = subparsers.add_parser(
        "prepare-bccd", help="Convert BCCD VOC annotations into YOLO format."
    )
    bccd_parser.add_argument("--source-dir", required=True)
    bccd_parser.add_argument("--output-dir", default="bccd-yolo")
    bccd_parser.add_argument("--train-ratio", type=float, default=0.7)
    bccd_parser.add_argument("--val-ratio", type=float, default=0.15)
    bccd_parser.add_argument("--seed", type=int, default=7)
    bccd_parser.add_argument("--max-images", type=int, default=0)
    bccd_parser.set_defaults(func=prepare_bccd)

    inspect_parser = subparsers.add_parser(
        "inspect-labels", help="Validate YOLO label files."
    )
    inspect_parser.add_argument("--data-yaml", required=True)
    inspect_parser.add_argument(
        "--split", default="train", choices=["train", "val", "test"]
    )
    inspect_parser.add_argument("--output-dir", default="object-detection-label-audit")
    inspect_parser.add_argument("--max-files", type=int, default=500)
    inspect_parser.add_argument("--example-rows", type=int, default=20)
    inspect_parser.add_argument("--allow-invalid", action="store_true")
    inspect_parser.set_defaults(func=inspect_labels)

    yolo_parser = subparsers.add_parser(
        "yolo", help="Build or run an Ultralytics command."
    )
    yolo_parser.add_argument(
        "--mode", default="train", choices=["train", "val", "test", "predict", "export"]
    )
    yolo_parser.add_argument("--model", default="yolo26n.pt")
    yolo_parser.add_argument("--data-yaml", default="medical-detection.yaml")
    yolo_parser.add_argument("--source", default="images/test")
    yolo_parser.add_argument("--imgsz", type=int, default=640)
    yolo_parser.add_argument("--epochs", type=int, default=100)
    yolo_parser.add_argument("--batch", default="auto")
    yolo_parser.add_argument("--seed", type=int, default=7)
    yolo_parser.add_argument("--project", default="runs/object-detection")
    yolo_parser.add_argument("--name", default="baseline")
    yolo_parser.add_argument("--export-format", default="onnx")
    yolo_parser.add_argument("--output-dir", default="object-detection-commands")
    yolo_parser.add_argument("--dry-run", action="store_true")
    yolo_parser.add_argument("--allow-missing-deps", action="store_true")
    yolo_parser.set_defaults(func=yolo_command)

    summary_parser = subparsers.add_parser(
        "summarize-runs", help="Summarize Ultralytics results.csv files."
    )
    summary_parser.add_argument("--runs-dir", default="runs/object-detection")
    summary_parser.add_argument("--output-dir", default="object-detection-summary")
    summary_parser.set_defaults(func=summarize_runs)

    plot_parser = subparsers.add_parser(
        "plot-metrics", help="Plot mAP50 from Ultralytics results.csv files."
    )
    plot_parser.add_argument("--runs-dir", default="runs/object-detection")
    plot_parser.add_argument("--output-dir", default="object-detection-plots")
    plot_parser.add_argument("--allow-missing-deps", action="store_true")
    plot_parser.add_argument("--allow-empty", action="store_true")
    plot_parser.set_defaults(func=plot_metrics)

    env_parser = subparsers.add_parser(
        "env-report", help="Record exact package versions."
    )
    env_parser.add_argument("--output-dir", default="object-detection-environment")
    env_parser.add_argument(
        "--packages",
        nargs="+",
        default=[
            "ultralytics",
            "torch",
            "torchvision",
            "numpy",
            "pyyaml",
            "pillow",
            "pydicom",
        ],
    )
    env_parser.set_defaults(func=environment_report)

    smoke_parser = subparsers.add_parser(
        "smoke", help="Run a dependency-light smoke check."
    )
    smoke_parser.add_argument("--output-dir", default="/tmp/adl-object-detection-smoke")
    smoke_parser.set_defaults(func=smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
