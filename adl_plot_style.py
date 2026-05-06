"""Standalone plot style used by public companion-code scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


PLOT_DPI = 180
INK = "#26323F"
MUTED = "#667085"
GRID = "#D9DEE7"
PANEL = "#F3F6FA"
BLUE = "#376D9E"
ORANGE = "#D9822B"
GREEN = "#4E9A51"
PURPLE = "#9A5FA4"
GRAY = "#6B7280"
RED = "#C85050"

QUALITATIVE_PALETTE = [BLUE, ORANGE, GREEN, PURPLE, GRAY, RED]
SPLIT_COLORS = {
    "train": BLUE,
    "training": BLUE,
    "validation": ORANGE,
    "val": ORANGE,
    "test": GREEN,
}
CONFUSION_LOW = "#F1F6FB"
CONFUSION_HIGH = BLUE


def apply_matplotlib_style(plt: Any) -> None:
    """Apply the shared style to Matplotlib figures."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlecolor": INK,
            "figure.dpi": 140,
            "savefig.dpi": PLOT_DPI,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )


def plot_theme(
    p9: Any, *, base_size: float = 10, legend_position: str = "right"
) -> Any:
    """Return the shared Plotnine theme for companion-code charts."""
    return p9.theme_minimal(base_size=base_size) + p9.theme(
        axis_text=p9.element_text(color=MUTED, size=8.5),
        axis_title=p9.element_text(color=INK, size=9.5),
        legend_position=legend_position,
        legend_text=p9.element_text(color=INK, size=8),
        legend_title=p9.element_text(color=INK, size=8.5),
        panel_background=p9.element_rect(fill="white", color="white"),
        panel_grid_major=p9.element_line(color=GRID, size=0.35),
        panel_grid_minor=p9.element_blank(),
        plot_background=p9.element_rect(fill="white", color="white"),
        plot_title=p9.element_text(color=INK, size=11, weight="bold", ha="center"),
        strip_background=p9.element_rect(fill=PANEL, color=GRID, size=0.5),
        strip_text=p9.element_text(color=INK, size=9.5, weight="bold"),
    )


def palette_for(keys: Iterable[Any]) -> dict[Any, str]:
    """Assign stable colors to the given keys."""
    palette: dict[Any, str] = {}
    fallback_index = 0
    for key in keys:
        if key in palette:
            continue
        normalized = str(key).lower()
        if normalized in SPLIT_COLORS:
            palette[key] = SPLIT_COLORS[normalized]
            continue
        palette[key] = QUALITATIVE_PALETTE[fallback_index % len(QUALITATIVE_PALETTE)]
        fallback_index += 1
    return palette


def repo_relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()
