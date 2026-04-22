from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np


def load_solver(solver_path: Path) -> Callable:
    module_name = f"solver_{solver_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, solver_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import solver from {solver_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "solve_tsp"):
        return wrap_solve_tsp(module.solve_tsp)
    if hasattr(module, "select_next_node"):
        return wrap_select_next_node(module.select_next_node)
    raise ValueError(f"Solver module {solver_path} does not expose solve_tsp or select_next_node")


def wrap_solve_tsp(solve_tsp: Callable) -> Callable:
    def solve(distance_matrix: np.ndarray, coordinates: list[tuple[float, float]] | None) -> list[int]:
        if not coordinates:
            raise ValueError("This solve_tsp algorithm requires coordinate or display data")

        result = solve_tsp(coordinates)
        if isinstance(result, tuple) and len(result) >= 2:
            return list(result[1])
        return list(result)

    return solve


def wrap_select_next_node(select_next_node: Callable) -> Callable:
    def solve(distance_matrix: np.ndarray, coordinates: list[tuple[float, float]] | None) -> list[int]:
        dimension = len(distance_matrix)
        if dimension == 0:
            return []

        current_node = 0
        destination_node = 0
        tour = [0]
        unvisited = set(range(1, dimension))

        while unvisited:
            candidate_nodes = np.array(
                sorted(unvisited, key=lambda node: (distance_matrix[current_node, node], node)),
                dtype=int,
            )
            next_node = int(select_next_node(current_node, destination_node, candidate_nodes, distance_matrix))
            if next_node not in unvisited:
                next_node = int(candidate_nodes[0])

            tour.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node

        return tour

    return solve
