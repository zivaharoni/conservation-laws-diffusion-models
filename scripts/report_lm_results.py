from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _collect_run_rows(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(results_root.rglob("metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        if "experiment_kind" not in payload or "benchmark" not in payload:
            continue
        benchmark = payload["benchmark"]
        eval_details = payload.get("eval_details", {})
        primary_split = str(payload.get("primary_split", "val"))
        primary_detail = eval_details.get(primary_split, {})
        rows.append(
            {
                "source": "run",
                "benchmark": benchmark.get("benchmark_name"),
                "tokenizer": benchmark.get("tokenizer_name"),
                "objective": payload.get("objective"),
                "run_id": payload.get("run_id"),
                "primary_split": primary_split,
                "nats_per_token": payload.get("ranking_metric_nats_per_token"),
                "curve_integral_nats": primary_detail.get("curve_integral_nats"),
                "dense_curve_integral_nats": primary_detail.get("dense_curve_integral_nats"),
                "bound_only_nats": None,
                "notes": str(metrics_path.parent),
            }
        )
    return rows


def _collect_literature_rows(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.is_file():
        return []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "source": "literature",
                    "benchmark": row.get("benchmark"),
                    "tokenizer": row.get("tokenizer"),
                    "objective": row.get("model_name", row.get("objective")),
                    "run_id": row.get("citation_key", ""),
                    "primary_split": row.get("split", ""),
                    "nats_per_token": row.get("nats_per_token", ""),
                    "curve_integral_nats": "",
                    "dense_curve_integral_nats": "",
                    "bound_only_nats": row.get("bound_only_nats", ""),
                    "notes": row.get("notes", ""),
                }
            )
        return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{lllllll}",
        r"\hline",
        r"Source & Benchmark & Tokenizer & Objective & Split & Nats/token & Bound-only \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    str(row.get("source", "")),
                    str(row.get("benchmark", "")),
                    str(row.get("tokenizer", "")),
                    str(row.get("objective", "")),
                    str(row.get("primary_split", "")),
                    str(row.get("nats_per_token", "")),
                    str(row.get("bound_only_nats", "")),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate LM run metrics and optional literature baselines.")
    parser.add_argument("--results-root", type=Path, default=Path("results/lm"))
    parser.add_argument("--literature-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=Path("results/lm/lm_summary.csv"))
    parser.add_argument("--output-tex", type=Path, default=Path("results/lm/lm_summary.tex"))
    args = parser.parse_args(argv)

    run_rows = _collect_run_rows(args.results_root)
    literature_rows = _collect_literature_rows(args.literature_csv)
    rows = run_rows + literature_rows
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_csv, rows)
    _write_latex(args.output_tex, rows)
    print(f"[report] wrote {len(rows)} rows to {args.output_csv} and {args.output_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
