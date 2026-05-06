#!/usr/bin/env python3
"""Check notebook companion-script lookup for public repo layout."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parent
PUBLIC_CHAPTER_CANDIDATE_RE = re.compile(r"base\s*/\s*chapter_name")
FIND_CHAPTER_CALL_RE = re.compile(
    r"find_(?:chapter|code)_dir\(\s*['\"]([^'\"]+\.py)['\"]\s*,\s*"
    r"['\"]([^'\"]+)['\"]\s*\)"
)
SCRIPT_CHAPTER_RE = re.compile(
    r"script_name\s*=\s*['\"]([^'\"]+\.py)['\"].{0,500}?"
    r"chapter_name\s*=\s*['\"]([^'\"]+)['\"]",
    re.DOTALL,
)


def tracked_notebooks() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--", "*.ipynb"],
        cwd=CODE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [CODE_ROOT / line for line in result.stdout.splitlines() if line]


def cell_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def code_text(cells: list[dict[str, Any]]) -> str:
    return "\n".join(
        cell_text(cell) for cell in cells if cell.get("cell_type") == "code"
    )


def companion_script_pairs(text: str) -> set[tuple[str, str]]:
    pairs = set(FIND_CHAPTER_CALL_RE.findall(text))
    pairs.update(SCRIPT_CHAPTER_RE.findall(text))
    return pairs


def notebook_failures(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    if not isinstance(cells, list):
        return ["notebook has no valid cells list"]

    text = code_text(cells)
    pairs = companion_script_pairs(text)
    if not pairs:
        return []

    failures: list[str] = []
    if not PUBLIC_CHAPTER_CANDIDATE_RE.search(text):
        failures.append(
            "companion-script lookup should include base / chapter_name "
            "for public repo root execution"
        )

    for script_name, chapter_name in sorted(pairs):
        script_path = CODE_ROOT / chapter_name / script_name
        if not script_path.exists():
            failures.append(
                f"{chapter_name}/{script_name} does not exist under code root"
            )

    return failures


def main() -> int:
    failures_by_path: dict[Path, list[str]] = {}
    for path in tracked_notebooks():
        failures = notebook_failures(path)
        if failures:
            failures_by_path[path.relative_to(CODE_ROOT)] = failures

    if not failures_by_path:
        return 0

    print("Notebook public path check failed:", file=sys.stderr)
    for path, failures in failures_by_path.items():
        print(f"  {path}", file=sys.stderr)
        for failure in failures:
            print(f"    - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
