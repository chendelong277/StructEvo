from __future__ import annotations

import random
import time
import zlib
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import calculate_tour_cost, load_instance, validate_tour
from .solvers import load_solver

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **_: Any):
        return iterable


RAW_COLUMNS = [
    "Algorithm",
    "Series",
    "Instance",
    "Dimension",
    "EdgeWeightType",
    "Run",
    "Seed",
    "OptCost",
    "Cost",
    "GapPct",
    "TimeSec",
    "Valid",
    "Message",
]


def run_experiment_for_paths(
    *,
    algorithm_paths: list[tuple[str, Path]],
    tsp_paths: list[Path],
    known_optima_path: Path,
    opt_tour_dir: Path,
    num_runs: int,
    processes: int,
    raw_output_dir: Path,
    resume: bool = True,
) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    if not tsp_paths:
        raise FileNotFoundError("No TSP instances were selected for this experiment")

    raw_frames: dict[str, pd.DataFrame] = {}
    for algorithm_name, solver_path in algorithm_paths:
        raw_csv_path = raw_output_dir / f"{algorithm_name}.csv"
        if resume and raw_csv_path.exists():
            print(f"[resume] {algorithm_name} <- {raw_csv_path}")
            raw_frames[algorithm_name] = pd.read_csv(raw_csv_path)
            continue

        print(f"[run] {algorithm_name}: {len(tsp_paths)} instances x {num_runs} runs")
        rows = _run_algorithm(
            algorithm_name=algorithm_name,
            solver_path=solver_path,
            tsp_paths=tsp_paths,
            known_optima_path=known_optima_path,
            opt_tour_dir=opt_tour_dir,
            num_runs=num_runs,
            processes=processes,
        )
        frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
        frame = frame.sort_values(["Dimension", "Series", "Instance", "Run"], kind="stable").reset_index(drop=True)
        frame.to_csv(raw_csv_path, index=False, float_format="%.6f")
        raw_frames[algorithm_name] = frame
        print(f"[saved] {raw_csv_path}")

    return raw_frames, tsp_paths


def _run_algorithm(
    *,
    algorithm_name: str,
    solver_path: Path,
    tsp_paths: list[Path],
    known_optima_path: Path,
    opt_tour_dir: Path,
    num_runs: int,
    processes: int,
) -> list[dict[str, Any]]:
    task_args = [
        {
            "algorithm_name": algorithm_name,
            "solver_path": str(solver_path),
            "tsp_path": str(tsp_path),
            "known_optima_path": str(known_optima_path),
            "opt_tour_dir": str(opt_tour_dir),
            "num_runs": num_runs,
        }
        for tsp_path in tsp_paths
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
    tsp_path = Path(task["tsp_path"])
    known_optima_path = Path(task["known_optima_path"])
    opt_tour_dir = Path(task["opt_tour_dir"])
    num_runs = int(task["num_runs"])

    try:
        solver = load_solver(solver_path)
        instance = load_instance(tsp_path, known_optima_path, opt_tour_dir)
        if instance.opt_cost is None:
            raise ValueError("Known optimum is missing for this instance")
    except Exception as exc:
        return [
            {
                "Algorithm": algorithm_name,
                "Series": tsp_path.stem,
                "Instance": tsp_path.stem,
                "Dimension": np.nan,
                "EdgeWeightType": "",
                "Run": run_index + 1,
                "Seed": np.nan,
                "OptCost": np.nan,
                "Cost": np.nan,
                "GapPct": np.nan,
                "TimeSec": np.nan,
                "Valid": False,
                "Message": str(exc),
            }
            for run_index in range(num_runs)
        ]

    rows: list[dict[str, Any]] = []
    base_seed = zlib.crc32(f"{algorithm_name}:{instance.name}".encode("utf-8")) & 0xFFFFFFFF

    for run_index in range(num_runs):
        seed = (base_seed + run_index) & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)

        start_time = time.perf_counter()
        cost = np.nan
        gap_pct = np.nan
        valid = False
        message = "OK"

        try:
            tour = solver(instance.distance_matrix, instance.coordinates)
            valid, message, normalized_tour = validate_tour(tour, instance.dimension)
            if valid and normalized_tour is not None:
                cost = calculate_tour_cost(normalized_tour, instance.evaluation_distance_matrix)
                gap_pct = (cost - instance.opt_cost) / instance.opt_cost * 100.0
        except Exception as exc:
            message = str(exc)

        elapsed = time.perf_counter() - start_time
        rows.append(
            {
                "Algorithm": algorithm_name,
                "Series": instance.series,
                "Instance": instance.name,
                "Dimension": instance.dimension,
                "EdgeWeightType": instance.edge_weight_type,
                "Run": run_index + 1,
                "Seed": int(seed),
                "OptCost": instance.opt_cost,
                "Cost": cost,
                "GapPct": gap_pct,
                "TimeSec": elapsed,
                "Valid": valid,
                "Message": message,
            }
        )

    return rows
