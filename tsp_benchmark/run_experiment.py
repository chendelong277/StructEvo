from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsp0324.analysis import analyze_experiment
from tsp0324.config import load_experiment_config, resolve_path
from tsp0324.dataset import build_instance_catalog
from tsp0324.executor import run_experiment_for_paths
from tsp0324.reporting import create_experiment_dir, write_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the refactored TSP batch experiment workflow.")
    parser.add_argument("--config", default="config/default_experiment.json", help="Path to the JSON experiment config.")
    parser.add_argument("--label", help="Override the experiment label.")
    parser.add_argument("--num-runs", type=int, help="Override the number of repeated runs per instance.")
    parser.add_argument("--processes", type=int, help="Override the process count.")
    parser.add_argument("--algorithms", help="Comma-separated algorithm names, for example EOHU,StructEvo.")
    parser.add_argument("--baseline", help="Override the baseline algorithm used for TEVC-style significance.")
    parser.add_argument("--max-dimension", type=int, help="Optional maximum dimension for selected instances.")
    parser.add_argument("--max-instances", type=int, help="Optional limit for the number of eligible instances.")
    parser.add_argument("--output-root", default="outputs", help="Root directory for new timestamped experiment folders.")
    parser.add_argument("--output-dir", help="Explicit experiment directory. Reuse this together with resume to continue a run.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing raw CSV files even if they exist.")
    args = parser.parse_args()

    config_path = resolve_path(ROOT_DIR, args.config)
    config = load_experiment_config(config_path).with_overrides(
        label=args.label,
        num_runs=args.num_runs,
        processes=args.processes,
        algorithm_names=_split_csv(args.algorithms),
        baseline_algorithm=args.baseline,
    )

    algorithm_paths = config.resolve_algorithms(ROOT_DIR)
    for algorithm_name, algorithm_path in algorithm_paths:
        if not algorithm_path.exists():
            raise FileNotFoundError(f"Algorithm '{algorithm_name}' was not found: {algorithm_path}")

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
    if args.output_dir:
        output_dir = resolve_path(ROOT_DIR, args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_root = resolve_path(ROOT_DIR, args.output_root)
        suffix = f"_nle{args.max_dimension}" if args.max_dimension else ""
        output_dir = create_experiment_dir(output_root, f"{config.label}{suffix}")

    dimension_counts = Counter(selected_catalog["SizeBucket"])
    edge_counts = Counter(selected_catalog["EdgeWeightType"])

    print(f"Output directory: {output_dir}")
    print(f"Algorithms: {', '.join(name for name, _ in algorithm_paths)}")
    print(f"Runs per instance: {config.num_runs}")
    print(f"Selected instances: {len(tsp_paths)}")
    print("Size buckets: " + ", ".join(f"{bucket}={dimension_counts[bucket]}" for bucket in sorted(dimension_counts)))
    print("Edge types: " + ", ".join(f"{edge_type}={edge_counts[edge_type]}" for edge_type in sorted(edge_counts)))

    raw_by_algorithm, instance_paths = run_experiment_for_paths(
        algorithm_paths=algorithm_paths,
        tsp_paths=tsp_paths,
        known_optima_path=known_optima_path,
        opt_tour_dir=opt_tour_dir,
        num_runs=config.num_runs,
        processes=config.processes,
        raw_output_dir=output_dir / "raw_runs",
        resume=not args.no_resume,
    )

    algorithm_order = [algorithm.name for algorithm in config.algorithms]
    bundle = analyze_experiment(
        raw_by_algorithm,
        algorithm_order=algorithm_order,
        baseline_algorithm=config.baseline_algorithm,
        alpha=config.alpha,
    )
    write_outputs(
        output_dir=output_dir,
        bundle=bundle,
        config_payload={
            **config.to_dict(),
            "max_dimension": args.max_dimension,
            "max_instances": args.max_instances,
        },
        algorithm_order=algorithm_order,
        baseline_algorithm=config.baseline_algorithm,
        source={"type": "live_run"},
        instance_catalog=instance_catalog,
        instance_paths=instance_paths,
    )

    print("Artifacts written:")
    print(f"  raw_runs/: {output_dir / 'raw_runs'}")
    print(f"  summary/:  {output_dir / 'summary'}")
    print(f"  tables/:   {output_dir / 'tables'}")
    return 0


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
