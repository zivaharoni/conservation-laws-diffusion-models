"""Simple curve-table export for later figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def export_curve_table(path: str | Path, rows: Iterable[tuple[float, float]], x_name: str = "x", y_name: str = "y") -> None:
    lines = [f"{x_name},{y_name}"]
    lines.extend(f"{float(x)},{float(y)}" for x, y in rows)
    Path(path).write_text("\n".join(lines) + "\n")
