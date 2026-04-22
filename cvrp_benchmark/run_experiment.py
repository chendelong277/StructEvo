from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cvrp0324.analysis import analyze_experiment
from cvrp0324.config import load_experiment_config, resolve_path
from cvrp0324.dataset import collect_vrp_paths, exclude_vrp_paths_by_stem
from cvrp0324.executor import run_experiment
from cvrp0324.executor import run_experiment_for_paths
from cvrp0324.reporting import create_experiment_dir, write_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CVRP batch experiments in the refactored cvrp0324 layout.")
    parser.add_argument("--config", default="config/default_experiment.json", help="Path to the JSON experiment config.")
    parser.add_argument("--label", help="Override the experiment label.")
    parser.add_argument("--num-runs", type=int, help="Override the number of repeated runs per instance.")
    parser.add_argument("--processes", type=int, help="Override the process count.")
    parser.add_argument("--families", help="Comma-separated instance families, for example A,B,X.")
    parser.add_argument("--algorithms", help="Comma-separated algorithm names, for example StructEvo,EOHU.")
    parser.add_argument("--baseline", help="Override the baseline algorithm used for TEVC-style significance.")
    parser.add_argument("--output-root", default="outputs", help="Root directory for new timestamped experiment folders.")
    parser.add_argument("--output-dir", help="Explicit experiment directory. Reuse this together with resume to continue a run.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing raw CSV files even if they exist.")
    args = parser.parse_args()

    config_path = resolve_path(ROOT_DIR, args.config)
    config = load_experiment_config(config_path)
    config = config.with_overrides(
        label=args.label,
        num_runs=args.num_runs,
        processes=args.processes,
        families=_split_csv(args.families),
        algorithm_names=_split_csv(args.algorithms),
        baseline_algorithm=args.baseline,
    )

    algorithm_paths = config.resolve_algorithms(ROOT_DIR)
    for algorithm_name, algorithm_path in algorithm_paths:
        if not algorithm_path.exists():
            raise FileNotFoundError(f"Algorithm '{algorithm_name}' was not found: {algorithm_path}")

    data_dir = config.resolve_data_dir(ROOT_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory was not found: {data_dir}")

    if args.output_dir:
        output_dir = resolve_path(ROOT_DIR, args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_root = resolve_path(ROOT_DIR, args.output_root)
        output_dir = create_experiment_dir(output_root, config.label)

    print(f"Output directory: {output_dir}")
    print(f"Families: {', '.join(config.families)}")
    print(f"Algorithms: {', '.join(name for name, _ in algorithm_paths)}")
    print(f"Runs per instance: {config.num_runs}")

    if config.excluded_instances:
        selected_paths = exclude_vrp_paths_by_stem(
            collect_vrp_paths(data_dir, config.families),
            config.excluded_instances,
        )
        raw_by_algorithm, instance_paths = run_experiment_for_paths(
            algorithm_paths=algorithm_paths,
            vrp_paths=selected_paths,
            num_runs=config.num_runs,
            processes=config.processes,
            raw_output_dir=output_dir / "raw_runs",
            resume=not args.no_resume,
        )
    else:
        raw_by_algorithm, instance_paths = run_experiment(
            algorithm_paths=algorithm_paths,
            data_dir=data_dir,
            families=config.families,
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
        config_payload=config.to_dict(),
        algorithm_order=algorithm_order,
        baseline_algorithm=config.baseline_algorithm,
        source={"type": "live_run"},
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
