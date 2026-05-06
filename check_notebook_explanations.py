#!/usr/bin/env python3
"""Check reader-facing explanation coverage in public notebooks."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parent
MIN_MARKDOWN_WORDS_PER_CODE_CELL = 25.0
MIN_MARKDOWN_CELLS_PER_CODE_CELL = 0.5
MIN_MARKDOWN_CELLS_WITH_CODE = 5
MAX_CONSECUTIVE_CODE_CELLS = 3
WORD_RE = re.compile(r"[A-Za-z0-9_']+")


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


def markdown_word_count(cells: list[dict[str, Any]]) -> int:
    return sum(
        len(WORD_RE.findall(cell_text(cell)))
        for cell in cells
        if cell.get("cell_type") == "markdown"
    )


def max_code_run(cells: list[dict[str, Any]]) -> int:
    longest = 0
    current = 0
    for cell in cells:
        if cell.get("cell_type") == "code":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def first_nonempty_cell_type(cells: list[dict[str, Any]]) -> str | None:
    for cell in cells:
        if cell_text(cell).strip():
            return str(cell.get("cell_type"))
    return None


def notebook_failures(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    if not isinstance(cells, list):
        return ["notebook has no valid cells list"]

    markdown_count = sum(1 for cell in cells if cell.get("cell_type") == "markdown")
    code_count = sum(1 for cell in cells if cell.get("cell_type") == "code")
    failures: list[str] = []

    first_cell_type = first_nonempty_cell_type(cells)
    if first_cell_type != "markdown":
        failures.append("first non-empty cell should be markdown")

    if code_count == 0:
        return failures

    if markdown_count < MIN_MARKDOWN_CELLS_WITH_CODE:
        failures.append(
            f"{markdown_count} markdown cells; "
            f"minimum is {MIN_MARKDOWN_CELLS_WITH_CODE} for notebooks with code"
        )

    words_per_code_cell = markdown_word_count(cells) / code_count
    if words_per_code_cell < MIN_MARKDOWN_WORDS_PER_CODE_CELL:
        failures.append(
            f"{words_per_code_cell:.1f} markdown words per code cell; "
            f"minimum is {MIN_MARKDOWN_WORDS_PER_CODE_CELL:.1f}"
        )

    markdown_cells_per_code_cell = markdown_count / code_count
    if markdown_cells_per_code_cell < MIN_MARKDOWN_CELLS_PER_CODE_CELL:
        failures.append(
            f"{markdown_cells_per_code_cell:.2f} markdown cells per code cell; "
            f"minimum is {MIN_MARKDOWN_CELLS_PER_CODE_CELL:.2f}"
        )

    longest_code_run = max_code_run(cells)
    if longest_code_run > MAX_CONSECUTIVE_CODE_CELLS:
        failures.append(
            f"{longest_code_run} consecutive code cells; "
            f"maximum is {MAX_CONSECUTIVE_CODE_CELLS}"
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

    print("Notebook explanation check failed:", file=sys.stderr)
    for path, failures in failures_by_path.items():
        print(f"  {path}", file=sys.stderr)
        for failure in failures:
            print(f"    - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
