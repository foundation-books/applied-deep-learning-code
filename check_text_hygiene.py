#!/usr/bin/env python3
"""Repository text hygiene checks for public companion-code releases."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def code_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "."],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted({Path(line) for line in result.stdout.splitlines() if line})


def has_extra_final_blank_line(data: bytes) -> bool:
    if b"\0" in data or not data.endswith((b"\n", b"\r")):
        return False
    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    lines = normalized.split(b"\n")
    return len(lines) >= 3 and lines[-2].strip(b" \t\f\v") == b""


def main() -> int:
    bad_paths: list[Path] = []
    for path in code_paths():
        if not path.is_file():
            continue
        if has_extra_final_blank_line(path.read_bytes()):
            bad_paths.append(path)

    if not bad_paths:
        return 0

    print("Files have an extra blank line at EOF:", file=sys.stderr)
    for path in bad_paths:
        print(f"  {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
