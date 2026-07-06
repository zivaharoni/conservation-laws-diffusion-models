from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALPHABET_SIZES = (8, 16, 32, 64, 128, 256, 512)


@dataclass(frozen=True)
class RunArtifact:
    metrics_path: Path
    run_dir: Path
    metrics: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_runs(results_root: Path) -> list[RunArtifact]:
    runs: list[RunArtifact] = []
    if not results_root.exists():
        return runs
    for metrics_path in sorted(results_root.rglob("metrics.json")):
        if "wandb" in metrics_path.parts:
            continue
        try:
            metrics = _read_json(metrics_path)
        except json.JSONDecodeError:
            continue
        runs.append(RunArtifact(metrics_path=metrics_path, run_dir=metrics_path.parent, metrics=metrics))
    return runs


def _as_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metric(metrics: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def _path_score(run: RunArtifact, preferred_parts: tuple[str, ...]) -> int:
    parts = set(run.metrics_path.parts)
    return sum(10 for part in preferred_parts if part in parts)


def _select_run(
    runs: list[RunArtifact],
    predicate: Callable[[RunArtifact], bool],
    *,
    label: str,
    preferred_parts: tuple[str, ...] = (),
    allow_missing: bool = False,
) -> RunArtifact | None:
    candidates = [run for run in runs if predicate(run)]
    if not candidates:
        if allow_missing:
            return None
        raise FileNotFoundError(f"No completed run artifact found for {label}.")
    candidates.sort(key=lambda run: (_path_score(run, preferred_parts), run.metrics_path.stat().st_mtime))
    return candidates[-1]


def _alphabet_size(run: RunArtifact) -> int | None:
    explicit = _as_int(_metric(run.metrics, "alphabet_size"))
    if explicit is not None:
        return explicit

    # Older QSC metrics did not persist alphabet_size. Prefer Hydra override
    # fragments over config names such as markov_channel_qsc_k128.
    path_text = "/".join(run.metrics_path.parts)
    for pattern in (
        r"data-alphabet-size-(\d+)",
        r"alphabet-size-(\d+)",
        r"[_-]k(\d+)(?:[/_]|$)",
    ):
        match = re.search(pattern, path_text)
        if match is not None:
            return int(match.group(1))
    return None


def _curve_path(run: RunArtifact, filename: str) -> Path:
    path = run.run_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Expected curve file {filename} next to {run.metrics_path}.")
    return path


def _copy_rows(src: Path, dst: Path, *, columns: dict[str, str] | None = None) -> None:
    rows = _read_csv(src)
    if columns is not None:
        rows = [{out_name: row[in_name] for out_name, in_name in columns.items()} for row in rows]
    _write_csv(dst, rows)


def _finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _cumulative_area(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    clean = [(float(x), float(y)) for x, y in points if math.isfinite(float(x)) and math.isfinite(float(y))]
    if not clean:
        return []
    clean.sort(key=lambda item: item[0])
    area = 0.0
    output = [(clean[0][0], 0.0)]
    for (x0, y0), (x1, y1) in zip(clean[:-1], clean[1:]):
        area += 0.5 * (y0 + y1) * (x1 - x0)
        output.append((x1, area))
    return output


def _format_coordinates(points: Iterable[tuple[Any, Any]]) -> str:
    lines = ["coordinates {"]
    for x, y in points:
        if _finite(x) and _finite(y):
            lines.append(f"({float(x):.10g},{float(y):.10g})")
    lines.append("};")
    return "\n".join(lines)


def _series_from_rows(rows: list[dict[str, str]], x_key: str, y_key: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        x = _as_float(row.get(x_key))
        y = _as_float(row.get(y_key))
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    return points


def export_markov_objective_data(
    *,
    runs: list[RunArtifact],
    output_root: Path,
    alphabet_sizes: tuple[int, ...],
    allow_missing: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for alphabet_size in alphabet_sizes:
        causal = _select_run(
            runs,
            lambda run, k=alphabet_size: _alphabet_size(run) == k and "eval_ce_nats_per_token" in run.metrics,
            label=f"Markov causal CE K={alphabet_size}",
            preferred_parts=("causal_ce", "markov"),
            allow_missing=allow_missing,
        )
        exit_run = _select_run(
            runs,
            lambda run, k=alphabet_size: _alphabet_size(run) == k and "eval_exit_area_nats" in run.metrics and str(run.metrics.get("family")) == "markov",
            label=f"Markov EXIT K={alphabet_size}",
            preferred_parts=("markov_scaling_exit",),
            allow_missing=allow_missing,
        )
        immse = _select_run(
            runs,
            lambda run, k=alphabet_size: _alphabet_size(run) == k and "model_area_nats" in run.metrics and "reference_entropy_nats" in run.metrics,
            label=f"Markov I-MMSE K={alphabet_size}",
            preferred_parts=("markov_scaling_immse",),
            allow_missing=allow_missing,
        )
        qsc = _select_run(
            runs,
            lambda run, k=alphabet_size: _alphabet_size(run) == k and (
                "model_ce_area_nats" in run.metrics or "model_bsc_area_nats" in run.metrics
            ),
            label=f"Markov QSC K={alphabet_size}",
            preferred_parts=("markov_channel_qsc_k128",),
            allow_missing=allow_missing,
        )
        selected = [run for run in (causal, exit_run, immse, qsc) if run is not None]
        if len(selected) < 4:
            continue
        entropy = _as_float(
            _metric(causal.metrics, "reference_entropy_nats_per_token")
            or _metric(exit_run.metrics, "reference_entropy_nats")
            or _metric(immse.metrics, "reference_entropy_nats")
            or _metric(qsc.metrics, "reference_entropy_nats")
        )
        causal_value = _as_float(_metric(causal.metrics, "eval_ce_nats_per_token"))
        exit_value = _as_float(_metric(exit_run.metrics, "eval_exit_area_nats"))
        immse_value = _as_float(_metric(immse.metrics, "model_area_nats"))
        qsc_value = _as_float(_metric(qsc.metrics, "model_ce_area_nats", "model_bsc_area_nats"))
        rows.append(
            {
                "K": alphabet_size,
                "entropy_nats": entropy,
                "causal_ce_nats": causal_value,
                "exit_area_nats": exit_value,
                "immse_area_nats": immse_value,
                "qsc": qsc_value,
            }
        )
        for objective, run in [("causal_ce", causal), ("exit", exit_run), ("immse", immse), ("qsc", qsc)]:
            source_rows.append({"artifact": f"markov_objective_k{alphabet_size}_{objective}", "source": str(run.metrics_path)})

    data_dir = output_root / "data" / "markov_synthetic"
    _write_csv(
        data_dir / "objective_estimates_d128.csv",
        rows,
        fieldnames=["K", "entropy_nats", "causal_ce_nats", "exit_area_nats", "immse_area_nats", "qsc"],
    )
    gap_rows = []
    for row in rows:
        entropy = max(abs(_as_float(row["entropy_nats"])), 1e-12)
        causal_gap = abs(_as_float(row["causal_ce_nats"]) - _as_float(row["entropy_nats"]))
        exit_gap = abs(_as_float(row["exit_area_nats"]) - _as_float(row["entropy_nats"]))
        immse_gap = abs(_as_float(row["immse_area_nats"]) - _as_float(row["entropy_nats"]))
        gap_rows.append(
            {
                "K": row["K"],
                "entropy_nats": row["entropy_nats"],
                "causal_abs_rel": causal_gap / entropy,
                "exit_rel": exit_gap / entropy,
                "immse_rel": immse_gap / entropy,
                "causal_gap": causal_gap,
                "exit_gap": exit_gap,
                "immse_gap": immse_gap,
            }
        )
    _write_csv(
        data_dir / "objective_gap_d128.csv",
        gap_rows,
        fieldnames=["K", "entropy_nats", "causal_abs_rel", "exit_rel", "immse_rel", "causal_gap", "exit_gap", "immse_gap"],
    )
    return source_rows


def export_markov_channel_data(*, runs: list[RunArtifact], output_root: Path, allow_missing: bool) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    data_dir = output_root / "data" / "markov_remote" / "channel_comparison"
    exit_run = _select_run(
        runs,
        lambda run: _alphabet_size(run) == 128 and "eval_exit_area_nats" in run.metrics and str(run.metrics.get("family")) == "markov",
        label="Markov channel EXIT K=128",
        preferred_parts=("markov_channel_exit_k128",),
        allow_missing=allow_missing,
    )
    qsc_run = _select_run(
        runs,
        lambda run: _alphabet_size(run) == 128 and ("model_ce_area_nats" in run.metrics or "model_bsc_area_nats" in run.metrics),
        label="Markov channel QSC K=128",
        preferred_parts=("markov_channel_qsc_k128",),
        allow_missing=allow_missing,
    )
    immse_run = _select_run(
        runs,
        lambda run: _alphabet_size(run) == 128 and "model_area_nats" in run.metrics and "reference_entropy_nats" in run.metrics,
        label="Markov channel I-MMSE K=128",
        preferred_parts=("markov_channel_immse_k128",),
        allow_missing=allow_missing,
    )
    if exit_run is not None:
        curve = _curve_path(exit_run, "exit_curve.csv")
        _copy_rows(
            curve,
            data_dir / "exit_overlay.csv",
            columns={"native_x": "s", "model_integrand": "model_exit", "reference_integrand": "true_exit"},
        )
        _copy_rows(curve, data_dir / "exit_k128.csv")
        sources.append({"artifact": "markov_channel_exit_k128", "source": str(exit_run.metrics_path)})
    if qsc_run is not None:
        curve = _curve_path(qsc_run, "exit_curve.csv")
        _copy_rows(
            curve,
            data_dir / "qsc_overlay.csv",
            columns={"native_x": "t", "model_integrand": "model_bsc_gexit", "reference_integrand": "ref_bsc_gexit"},
        )
        _copy_rows(curve, data_dir / "qsc_k128.csv")
        sources.append({"artifact": "markov_channel_qsc_k128", "source": str(qsc_run.metrics_path)})
    if immse_run is not None:
        curve = _curve_path(immse_run, "immse_curve.csv")
        _copy_rows(curve, data_dir / "immse_k128.csv")
        sources.append({"artifact": "markov_channel_immse_k128", "source": str(immse_run.metrics_path)})
    summary_rows = []
    for objective, run, area_key in [
        ("exit", exit_run, "eval_exit_area_nats"),
        ("qsc", qsc_run, "model_ce_area_nats"),
        ("immse", immse_run, "model_area_nats"),
    ]:
        if run is None:
            continue
        summary_rows.append(
            {
                "objective": objective,
                "K": 128,
                "reference_entropy_nats": _metric(run.metrics, "reference_entropy_nats", "reference_entropy_nats_per_token"),
                "model_area_nats": _metric(run.metrics, area_key, "model_bsc_area_nats"),
                "source": str(run.metrics_path),
            }
        )
    if summary_rows:
        _write_csv(data_dir / "summary.csv", summary_rows)
    return sources


def _copy_objective_curve(run: RunArtifact, output_dir: Path, objective: str) -> Path:
    filenames = {"exit": "exit_curve.csv", "qsc": "qsc_curve.csv", "immse": "immse_curve.csv"}
    destination = output_dir / f"{objective}_curve.csv"
    _copy_rows(_curve_path(run, filenames[objective]), destination)
    return destination


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def export_latest_conservation_data(
    *,
    runs: list[RunArtifact],
    output_root: Path,
    family: str,
    allow_missing: bool,
) -> list[dict[str, Any]]:
    if family == "text8":
        data_dir = output_root / "data" / "text8_conservation_curves" / "latest"
        selectors = {
            "exit": ("lm_exit", "text8_exit"),
            "qsc": ("lm_qsc", "text8_qsc"),
            "immse": ("lm_immse", "text8_immse"),
        }
    elif family == "cifar":
        data_dir = output_root / "data" / "cifar_conservation_curves" / "latest"
        selectors = {
            "exit": ("img_exit", "cifar10_rgb8_unet_exit"),
            "qsc": ("img_qsc", "cifar10_rgb8_unet_qsc"),
            "immse": ("img_immse", "cifar10_rgb8_unet_immse"),
        }
    else:
        raise ValueError(f"Unsupported conservation-data family {family!r}.")

    sources: list[dict[str, Any]] = []
    selected: dict[str, RunArtifact] = {}
    data_dir.mkdir(parents=True, exist_ok=True)
    _unlink_if_exists(data_dir / "manifest.json")
    for objective, (experiment_kind, preferred_part) in selectors.items():
        run = _select_run(
            runs,
            lambda candidate, kind=experiment_kind: str(candidate.metrics.get("experiment_kind")) == kind,
            label=f"{family} {objective}",
            preferred_parts=(preferred_part,),
            allow_missing=allow_missing,
        )
        if run is None:
            _unlink_if_exists(data_dir / f"{objective}_curve.csv")
            _unlink_if_exists(data_dir / f"{objective}_metrics.json")
            continue
        selected[objective] = run
        _copy_objective_curve(run, data_dir, objective)
        _write_json(data_dir / f"{objective}_metrics.json", run.metrics)
        sources.append({"artifact": f"{family}_{objective}", "source": str(run.metrics_path)})
    if selected:
        _write_json(
            data_dir / "manifest.json",
            {
                "family": family,
                "sources": {objective: str(run.metrics_path) for objective, run in selected.items()},
            },
        )
    return sources


def _generate_conservation_tikz(*, data_dir: Path, relative_data_dir: str, title_prefix: str = "") -> str:
    exit_rows = _read_csv(data_dir / "exit_curve.csv")
    qsc_rows = _read_csv(data_dir / "qsc_curve.csv")
    immse_rows = _read_csv(data_dir / "immse_curve.csv")
    exit_points = _series_from_rows(exit_rows, "s", "exit_gap_nats")
    qsc_points = [(1.0 - x, y) for x, y in _series_from_rows(qsc_rows, "t", "qsc_gexit_nats")]
    immse_points = _series_from_rows(immse_rows, "snr", "half_mmse")
    exit_area = _cumulative_area(exit_points)
    qsc_area = _cumulative_area(qsc_points)
    immse_progress = [(idx / max(len(immse_points) - 1, 1), y) for idx, (_x, y) in enumerate(immse_points)]
    immse_area = _cumulative_area(immse_progress)
    return "\n".join(
        [
            "% Auto-generated by scripts/export_paper_artifacts.py",
            r"\begin{tikzpicture}",
            r"\begin{groupplot}[",
            r"group style={group size=3 by 1, horizontal sep=0.95cm},",
            r"width=0.38\linewidth,",
            r"height=0.35\linewidth,",
            r"grid=both,",
            r"major grid style={draw=gray!25},",
            r"minor grid style={draw=gray!15},",
            r"tick label style={font=\footnotesize},",
            r"label style={font=\footnotesize},",
            r"title style={font=\footnotesize},",
            r"legend style={draw=none, fill=none, font=\scriptsize},",
            r"]",
            r"\nextgroupplot[xlabel={Mask rate $t$ / replace rate $1-t$}, ylabel={CE derivative}, xmin=0, xmax=1, legend pos=north west]",
            rf"\addplot+[line width=1.15pt, color=teal!70!black, mark=none] table [x=s, y=exit_gap_nats, col sep=comma] {{{relative_data_dir}/exit_curve.csv}};",
            r"\addlegendentry{Masked}",
            rf"\addplot+[line width=1.15pt, color=blue!70!black, mark=none] table [x expr=1-\thisrow{{t}}, y=qsc_gexit_nats, col sep=comma] {{{relative_data_dir}/qsc_curve.csv}};",
            r"\addlegendentry{Uniform}",
            r"\nextgroupplot[xlabel={SNR $t$}, ylabel={Half-MSE}, xmode=log, ymin=0, legend pos=north east]",
            rf"\addplot+[line width=1.15pt, color=red!75!black, mark=none] table [x=snr, y=half_mmse, col sep=comma] {{{relative_data_dir}/immse_curve.csv}};",
            r"\addlegendentry{Gaussian}",
            r"\nextgroupplot[xlabel={Noise progress}, ylabel={Cumulative area}, xmin=0, xmax=1, ymin=0, legend pos=north west]",
            r"\addplot+[line width=1.15pt, color=teal!70!black, mark=none]",
            _format_coordinates(exit_area),
            r"\addlegendentry{Masked}",
            r"\addplot+[line width=1.15pt, color=blue!70!black, mark=none]",
            _format_coordinates(qsc_area),
            r"\addlegendentry{Uniform}",
            r"\addplot+[line width=1.15pt, color=red!75!black, mark=none]",
            _format_coordinates(immse_area),
            r"\addlegendentry{Gaussian}",
            r"\end{groupplot}",
            r"\end{tikzpicture}",
            "",
        ]
    )


def generate_conservation_tikz(*, output_root: Path, family: str, allow_missing: bool) -> list[dict[str, Any]]:
    if family == "text8":
        data_dir = output_root / "data" / "text8_conservation_curves" / "latest"
        tex_path = output_root / "tikz" / "exp_text8_eval_curves.tex"
        relative = "shared/figures/data/text8_conservation_curves/latest"
    elif family == "cifar":
        data_dir = output_root / "data" / "cifar_conservation_curves" / "latest"
        tex_path = output_root / "tikz" / "exp_cifar_conservation_curves.tex"
        relative = "shared/figures/data/cifar_conservation_curves/latest"
    else:
        raise ValueError(f"Unsupported conservation-TikZ family {family!r}.")
    required = [data_dir / "exit_curve.csv", data_dir / "qsc_curve.csv", data_dir / "immse_curve.csv"]
    manifest_path = data_dir / "manifest.json"
    manifest_missing = []
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        sources = manifest.get("sources", {})
        manifest_missing = [objective for objective in ("exit", "qsc", "immse") if objective not in sources]
    else:
        manifest_missing = ["manifest.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing or manifest_missing:
        if allow_missing:
            _unlink_if_exists(tex_path)
            return []
        raise FileNotFoundError(
            f"Cannot generate {family} conservation TikZ; missing files={missing} manifest_entries={manifest_missing}."
        )
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(_generate_conservation_tikz(data_dir=data_dir, relative_data_dir=relative), encoding="utf-8")
    return [{"artifact": f"{family}_conservation_tikz", "source": str(data_dir)}]


def copy_cifar_suite_panels(*, results_root: Path, output_root: Path) -> list[dict[str, Any]]:
    suite_root = results_root / "img" / "cifar_suite"
    if not suite_root.exists():
        return []
    pngs = [path for path in suite_root.rglob("*.png") if "wandb" not in path.parts]
    selected = sorted(pngs, key=lambda path: path.stat().st_mtime)[-12:]
    dest_dir = output_root / "data" / "cifar_suite" / "generated"
    sources: list[dict[str, Any]] = []
    for path in selected:
        dest_dir.mkdir(parents=True, exist_ok=True)
        destination = dest_dir / path.name
        shutil.copy2(path, destination)
        sources.append({"artifact": f"cifar_suite/{destination.name}", "source": str(path)})
    return sources


def parse_alphabet_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes:
        raise ValueError("--alphabet-sizes must contain at least one integer.")
    return sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export run artifacts into the paper figure-data layout.")
    parser.add_argument("--results-root", type=Path, default=APP_ROOT / "results")
    parser.add_argument("--paper-root", type=Path, default=APP_ROOT.parent / "paper")
    parser.add_argument("--output-root", type=Path, default=None, help="Defaults to PAPER_ROOT/shared/figures.")
    parser.add_argument("--alphabet-sizes", default=",".join(str(value) for value in DEFAULT_ALPHABET_SIZES))
    parser.add_argument("--allow-missing", action="store_true", help="Write the artifacts that can be reconstructed and skip missing runs.")
    parser.add_argument("--skip-cifar-suite-panels", action="store_true")
    args = parser.parse_args(argv)

    results_root = args.results_root.expanduser().resolve()
    paper_root = args.paper_root.expanduser().resolve()
    output_root = (args.output_root.expanduser().resolve() if args.output_root is not None else paper_root / "shared" / "figures")
    alphabet_sizes = parse_alphabet_sizes(args.alphabet_sizes)
    runs = _load_runs(results_root)
    if not runs and not args.allow_missing:
        raise FileNotFoundError(f"No metrics.json files found under {results_root}.")

    sources: list[dict[str, Any]] = []
    sources.extend(export_markov_objective_data(runs=runs, output_root=output_root, alphabet_sizes=alphabet_sizes, allow_missing=args.allow_missing))
    sources.extend(export_markov_channel_data(runs=runs, output_root=output_root, allow_missing=args.allow_missing))
    sources.extend(export_latest_conservation_data(runs=runs, output_root=output_root, family="text8", allow_missing=args.allow_missing))
    sources.extend(export_latest_conservation_data(runs=runs, output_root=output_root, family="cifar", allow_missing=args.allow_missing))
    sources.extend(generate_conservation_tikz(output_root=output_root, family="text8", allow_missing=args.allow_missing))
    sources.extend(generate_conservation_tikz(output_root=output_root, family="cifar", allow_missing=args.allow_missing))
    if not args.skip_cifar_suite_panels:
        sources.extend(copy_cifar_suite_panels(results_root=results_root, output_root=output_root))

    _write_json(
        output_root / "data" / "paper_export_manifest.json",
        {
            "results_root": str(results_root),
            "paper_root": str(paper_root),
            "output_root": str(output_root),
            "artifact_count": len(sources),
            "sources": sources,
        },
    )
    print(f"[paper-export] wrote {len(sources)} artifact mappings under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
