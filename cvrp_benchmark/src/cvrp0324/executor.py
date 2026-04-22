from __future__ import annotations

import math
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import calculate_total_distance, collect_vrp_paths, load_instance, validate_routes
from .solvers import load_solver

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **_: Any):
        return iterable


RAW_COLUMNS = [
    "Algorithm",
    "Family",
    "Instance",
    "Run",
    "OptCost",
    "Cost",
    "GapPct",
    "TimeSec",
    "Valid",
    "Message",
]


def run_experiment(
    *,
    algorithm_paths: list[tuple[str, Path]],
    data_dir: Path,
    families: tuple[str, ...],
    num_runs: int,
    processes: int,
    raw_output_dir: Path,
    resume: bool = True,
) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    vrp_paths = collect_vrp_paths(data_dir, families)
    return run_experiment_for_paths(
        algorithm_paths=algorithm_paths,
        vrp_paths=vrp_paths,
        num_runs=num_runs,
        processes=processes,
        raw_output_dir=raw_output_dir,
        resume=resume,
    )


def run_experiment_for_paths(
    *,
    algorithm_paths: list[tuple[str, Path]],
    vrp_paths: list[Path],
    num_runs: int,
    processes: int,
    raw_output_dir: Path,
    resume: bool = True,
) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    if not vrp_paths:
        raise FileNotFoundError("No VRP instances were selected for this experiment")

    raw_frames: dict[str, pd.DataFrame] = {}
    for algorithm_name, solver_path in algorithm_paths:
        raw_csv_path = raw_output_dir / f"{algorithm_name}.csv"
        if resume and raw_csv_path.exists():
            print(f"[resume] {algorithm_name} <- {raw_csv_path}")
            raw_frames[algorithm_name] = pd.read_csv(raw_csv_path)
            continue

        print(f"[run] {algorithm_name}: {len(vrp_paths)} instances x {num_runs} runs")
        rows = _run_algorithm(
            algorithm_name=algorithm_name,
            solver_path=solver_path,
            vrp_paths=vrp_paths,
            num_runs=num_runs,
            processes=processes,
        )
        frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
        frame = frame.sort_values(["Family", "Instance", "Run"], kind="stable").reset_index(drop=True)
        frame.to_csv(raw_csv_path, index=False, float_format="%.6f")
        raw_frames[algorithm_name] = frame
        print(f"[saved] {raw_csv_path}")

    return raw_frames, vrp_paths


def _run_algorithm(
    *,
    algorithm_name: str,
    solver_path: Path,
    vrp_paths: list[Path],
    num_runs: int,
    processes: int,
) -> list[dict[str, Any]]:
    task_args = [
        {
            "algorithm_name": algorithm_name,
            "solver_path": str(solver_path),
            "vrp_path": str(vrp_path),
            "num_runs": num_runs,
        }
        for vrp_path in vrp_paths
    ]
    worker_count = max(1, min(processes, len(task_args), cpu_count() or 1))

    rows: list[dict[str, Any]] = []
    with Pool(processes=worker_count) as pool:
        iterator = pool.imap_unordered(_run_single_instance, task_args)
        for chunk in tqdm(iterator, total=len(task_args), desc=algorithm_name, unit="instance"):
            rows.extend(chunk)
    return rows


def _run_single_instance(task: dict[str, Any]) -> list[dict[str, Any]]:
    algorithm_name = task["algorithm_name"]
    solver_path = Path(task["solver_path"])
    vrp_path = Path(task["vrp_path"])
    num_runs = int(task["num_runs"])

    try:
        loaded_solver = load_solver(solver_path)
        instance = load_instance(vrp_path)
    except Exception as exc:
        return [
            {
                "Algorithm": algorithm_name,
                "Family": vrp_path.parent.name,
                "Instance": vrp_path.stem,
                "Run": run_index + 1,
                "OptCost": math.nan,
                "Cost": math.nan,
                "GapPct": math.nan,
                "TimeSec": math.nan,
                "Valid": False,
                "Message": str(exc),
            }
            for run_index in range(num_runs)
        ]

    rows: list[dict[str, Any]] = []
    for run_index in range(num_runs):
        start_time = time.perf_counter()
        cost = math.nan
        gap_pct = math.nan
        valid = False
        message = "OK"

        try:
            solver_distance_matrix = (
                instance.constructive_distance_matrix
                if loaded_solver.kind == "select_next_node"
                else instance.distance_matrix
            )
            routes = loaded_solver.solve(
                solver_distance_matrix,
                instance.demands,
                instance.capacity,
                instance.depot_index,
            )
            valid, message = validate_routes(
                routes,
                instance.demands,
                instance.capacity,
                instance.depot_index,
            )
            if valid:
                cost = float(calculate_total_distance(routes, instance.evaluation_distance_matrix))
                if instance.opt_cost:
                    gap_pct = (cost - instance.opt_cost) / instance.opt_cost * 100.0
        except Exception as exc:
            message = str(exc)

        elapsed = time.perf_counter() - start_time
        rows.append(
            {
                "Algorithm": algorithm_name,
                "Family": instance.family,
                "Instance": instance.name,
                "Run": run_index + 1,
                "OptCost": instance.opt_cost if instance.opt_cost is not None else math.nan,
                "Cost": cost,
                "GapPct": gap_pct,
                "TimeSec": elapsed,
                "Valid": valid,
                "Message": message,
            }
        )

    return rows
