from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsp0324.analysis import analyze_experiment
from tsp0324.config import load_experiment_config, resolve_path
from tsp0324.dataset import build_instance_catalog
from tsp0324.executor import run_experiment_for_paths
from tsp0324.reporting import create_experiment_dir, write_outputs


DEFAULT_BUDGETS = (50, 100, 200, 500, 1000, 2000, 5000, 10000)
DEFAULT_REUSE_10000 = "outputs/20260330_165629_benchmark_le150_fullset_normfix_nle150"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a StructEvo max-evals sweep on TSPLib instances with dimension <= 150."
    )
    parser.add_argument("--config", default="config/default_experiment.json", help="Path to the JSON experiment config.")
    parser.add_argument("--label", default="structevo_budget_sweep_le150", help="Experiment label.")
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS), help="Comma-separated max-evals values.")
    parser.add_argument("--max-dimension", type=int, default=150, help="Maximum instance dimension to include.")
    parser.add_argument("--num-runs", type=int, help="Override the number of repeated runs per instance.")
    parser.add_argument("--processes", type=int, help="Override the process count.")
    parser.add_argument("--max-instances", type=int, help="Optional limit for the number of eligible instances.")
    parser.add_argument("--output-root", default="outputs", help="Root directory for the sweep output folder.")
    parser.add_argument(
        "--reuse-budget-10000",
        default=DEFAULT_REUSE_10000,
        help="Existing output directory to reuse for max_evals=10000. Leave empty to rerun it.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing raw CSV files inside newly created budget folders.")
    args = parser.parse_args()

    budgets = _parse_budgets(args.budgets)
    if not budgets:
        raise ValueError("At least one max-evals value is required")

    config_path = resolve_path(ROOT_DIR, args.config)
    config = load_experiment_config(config_path).with_overrides(
        label=args.label,
        num_runs=args.num_runs,
        processes=args.processes,
        algorithm_names=["StructEvo"],
        baseline_algorithm="StructEvo",
    )

    algorithm_paths = config.resolve_algorithms(ROOT_DIR)
    if len(algorithm_paths) != 1 or algorithm_paths[0][0] != "StructEvo":
        raise ValueError("The sweep expects exactly one selected algorithm: StructEvo")
    _, solver_path = algorithm_paths[0]
    if not solver_path.exists():
        raise FileNotFoundError(f"StructEvo solver was not found: {solver_path}")

    data_dir = config.resolve_data_dir(ROOT_DIR)
    known_optima_path = resolve_path(ROOT_DIR, "data/known_optima.txt")
    opt_tour_dir = resolve_path(ROOT_DIR, "data/opt_tour")
    instance_catalog = build_instance_catalog(
        data_dir=data_dir,
        known_optima_path=known_optima_path,
        opt_tour_dir=opt_tour_dir,
        max_dimension=args.max_dimension,
        max_instances=args.max_instances,
        require_geometry=True,
    )
    selected_catalog = instance_catalog[instance_catalog["Eligible"]].copy()
    if selected_catalog.empty:
        raise FileNotFoundError("No eligible TSP instances were selected")
    tsp_paths = [Path(path_str) for path_str in selected_catalog["Path"].tolist()]

    output_root = resolve_path(ROOT_DIR, args.output_root)
    sweep_dir = create_experiment_dir(output_root, f"{config.label}_nle{args.max_dimension}")
    print(f"Sweep output directory: {sweep_dir}")
    print(f"Selected instances: {len(tsp_paths)}")
    print(f"Budgets: {', '.join(str(value) for value in budgets)}")

    reuse_10000_dir = None
    if args.reuse_budget_10000:
        reuse_10000_dir = resolve_path(ROOT_DIR, args.reuse_budget_10000)
        if not (reuse_10000_dir / "summary" / "instance_summary_long.csv").exists():
            print(f"[warn] Reuse directory not found or incomplete: {reuse_10000_dir}. max_evals=10000 will be rerun.")
            reuse_10000_dir = None

    rows: list[dict[str, object]] = []
    budget_sources: dict[int, str] = {}

    for budget in budgets:
        if budget == 10000 and reuse_10000_dir is not None:
            budget_dir = reuse_10000_dir
            print(f"[reuse] max_evals={budget} <- {budget_dir}")
            budget_sources[budget] = str(budget_dir)
        else:
            budget_dir = sweep_dir / f"budget_{budget}"
            budget_dir.mkdir(parents=True, exist_ok=True)
            print(f"[run] max_evals={budget} -> {budget_dir}")
            _run_single_budget(
                budget=budget,
                config=config,
                algorithm_paths=algorithm_paths,
                tsp_paths=tsp_paths,
                known_optima_path=known_optima_path,
                opt_tour_dir=opt_tour_dir,
                instance_catalog=instance_catalog,
                output_dir=budget_dir,
                max_dimension=args.max_dimension,
                max_instances=args.max_instances,
                resume=not args.no_resume,
            )
            budget_sources[budget] = str(budget_dir)

        rows.append(_summarize_budget_output(budget, budget_dir))

    summary_frame = pd.DataFrame(rows).sort_values("MaxEvals", kind="stable").reset_index(drop=True)
    summary_dir = sweep_dir / "summary"
    tables_dir = sweep_dir / "tables"
    figures_dir = sweep_dir / "figures"
    meta_dir = sweep_dir / "meta"
    summary_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    summary_frame.to_csv(summary_dir / "budget_sweep_summary.csv", index=False, float_format="%.6f")
    with pd.ExcelWriter(summary_dir / "budget_sweep_summary.xlsx", engine="openpyxl") as writer:
        summary_frame.to_excel(writer, sheet_name="budget_sweep", index=False)

    (tables_dir / "budget_sweep_table.tex").write_text(_render_budget_table(summary_frame), encoding="utf-8")
    figure_path = figures_dir / "structevo_tsp_budget_sweep.pdf"
    _plot_budget_sweep(summary_frame, figure_path)

    manifest = {
        "label": config.label,
        "budgets": budgets,
        "num_runs": config.num_runs,
        "processes": config.processes,
        "max_dimension": args.max_dimension,
        "max_instances": args.max_instances,
        "selected_instance_count": len(tsp_paths),
        "budget_sources": budget_sources,
    }
    (meta_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Artifacts written:")
    print(f"  summary: {summary_dir}")
    print(f"  tables:  {tables_dir}")
    print(f"  figures: {figures_dir}")
    return 0


def _parse_budgets(raw_value: str) -> list[int]:
    budgets = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value < 1:
            raise ValueError(f"Invalid max-evals value: {chunk}")
        budgets.append(value)
    unique_sorted = sorted(set(budgets))
    return unique_sorted


def _run_single_budget(
    *,
    budget: int,
    config,
    algorithm_paths: list[tuple[str, Path]],
    tsp_paths: list[Path],
    known_optima_path: Path,
    opt_tour_dir: Path,
    instance_catalog: pd.DataFrame,
    output_dir: Path,
    max_dimension: int,
    max_instances: int | None,
    resume: bool,
) -> None:
    old_value = os.environ.get("STRUCTEVO_MAX_EVALS")
    os.environ["STRUCTEVO_MAX_EVALS"] = str(budget)
    try:
        raw_by_algorithm, instance_paths = run_experiment_for_paths(
            algorithm_paths=algorithm_paths,
            tsp_paths=tsp_paths,
            known_optima_path=known_optima_path,
            opt_tour_dir=opt_tour_dir,
            num_runs=config.num_runs,
            processes=config.processes,
            raw_output_dir=output_dir / "raw_runs",
            resume=resume,
        )
        bundle = analyze_experiment(
            raw_by_algorithm,
            algorithm_order=["StructEvo"],
            baseline_algorithm="StructEvo",
            alpha=config.alpha,
        )
        write_outputs(
            output_dir=output_dir,
            bundle=bundle,
            config_payload={
                **config.to_dict(),
                "max_dimension": max_dimension,
                "max_instances": max_instances,
                "structevo_max_evals": budget,
            },
            algorithm_order=["StructEvo"],
            baseline_algorithm="StructEvo",
            source={"type": "live_run", "max_dimension": max_dimension, "structevo_max_evals": budget},
            instance_catalog=instance_catalog,
            instance_paths=instance_paths,
        )
    finally:
        if old_value is None:
            os.environ.pop("STRUCTEVO_MAX_EVALS", None)
        else:
            os.environ["STRUCTEVO_MAX_EVALS"] = old_value


def _summarize_budget_output(budget: int, output_dir: Path) -> dict[str, object]:
    overall_path = output_dir / "summary" / "overall_summary.csv"
    instance_path = output_dir / "summary" / "instance_summary_long.csv"
    overall = pd.read_csv(overall_path)
    instance_summary = pd.read_csv(instance_path)

    overall_row = overall.loc[overall["Algorithm"] == "StructEvo"].iloc[0]
    struct_instances = instance_summary.loc[instance_summary["Algorithm"] == "StructEvo"].copy()
    runtime_values = struct_instances["MeanTimeSec"].astype(float).to_numpy()
    gap_values = struct_instances["MeanGapPct"].astype(float).to_numpy()

    return {
        "MaxEvals": int(budget),
        "SolvedInstances": int(overall_row["SolvedInstances"]),
        "MeanGapPct": float(overall_row["MeanGapPct"]),
        "StdGapPctAcrossInstances": float(overall_row["StdGapPctAcrossInstances"]),
        "MeanBestGapPct": float(overall_row["MeanBestGapPct"]),
        "MeanTimeSec": float(overall_row["MeanTimeSec"]),
        "StdTimeSecAcrossInstances": float(np.std(runtime_values, ddof=0)),
        "MinGapPct": float(np.min(gap_values)),
        "MaxGapPct": float(np.max(gap_values)),
        "OutputDir": str(output_dir),
    }


def _render_budget_table(summary_frame: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Effect of the maximum evaluation budget on StructEvo over the 23 TSPLib instances with at most 150 nodes. Each entry reports the mean across instances, and $\pm$ denotes the standard deviation across instances. Lower is better.}",
        r"\label{tab:tsp_structevo_budget_sweep}",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"Max evals & Mean gap (\%) & Mean runtime (s) & Solved instances \\",
        r"\midrule",
    ]
    for row in summary_frame.itertuples(index=False):
        lines.append(
            f"{int(row.MaxEvals)} & "
            f"{row.MeanGapPct:.3f} $\\pm$ {row.StdGapPctAcrossInstances:.3f} & "
            f"{row.MeanTimeSec:.3f} $\\pm$ {row.StdTimeSecAcrossInstances:.3f} & "
            f"{int(row.SolvedInstances)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _plot_budget_sweep(summary_frame: pd.DataFrame, output_path: Path) -> None:
    budgets = summary_frame["MaxEvals"].to_numpy(dtype=float)
    mean_gap = summary_frame["MeanGapPct"].to_numpy(dtype=float)
    std_gap = summary_frame["StdGapPctAcrossInstances"].to_numpy(dtype=float)
    mean_time = summary_frame["MeanTimeSec"].to_numpy(dtype=float)
    std_time = summary_frame["StdTimeSecAcrossInstances"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.0), sharex=True, constrained_layout=True)

    axes[0].errorbar(budgets, mean_gap, yerr=std_gap, color="#1f4e79", marker="o", linewidth=1.8, capsize=3)
    axes[0].set_ylabel("Mean gap (%)")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0].set_title("StructEvo on TSPLib (<=150 nodes): budget vs. solution quality")

    axes[1].errorbar(budgets, mean_time, yerr=std_time, color="#b35c1e", marker="s", linewidth=1.8, capsize=3)
    axes[1].set_ylabel("Mean runtime (s)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Maximum evaluations")
    axes[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(budgets)
        axis.set_xticklabels([str(int(value)) for value in budgets], rotation=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
