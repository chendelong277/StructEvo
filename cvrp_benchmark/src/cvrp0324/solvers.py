from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class LoadedSolver:
    solve: Callable
    kind: str


def load_solver(solver_path: Path) -> LoadedSolver:
    module_name = f"solver_{solver_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, solver_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import solver from {solver_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "solve_vrp"):
        return LoadedSolver(solve=module.solve_vrp, kind="solve_vrp")
    if hasattr(module, "select_next_node"):
        return LoadedSolver(solve=wrap_select_next_node(module.select_next_node), kind="select_next_node")
    raise ValueError(f"Solver module {solver_path} does not expose solve_vrp or select_next_node")


def wrap_select_next_node(select_next_node: Callable) -> Callable:
    def solve_vrp(distance_matrix, demands, capacity, depot_index=0):
        node_count = len(distance_matrix)
        routes: list[list[int]] = []
        unvisited = set(range(node_count)) - {depot_index}
        demands_array = np.asarray(demands)

        while unvisited:
            current_node = depot_index
            remaining_capacity = capacity
            route = [depot_index]

            while unvisited:
                candidate_nodes = np.array(sorted(unvisited), dtype=int)
                next_node = int(
                    select_next_node(
                        current_node,
                        depot_index,
                        candidate_nodes,
                        remaining_capacity,
                        demands_array,
                        distance_matrix,
                    )
                )

                if next_node == depot_index:
                    if len(route) > 1:
                        break
                    feasible_nodes = [node for node in candidate_nodes if demands_array[node] <= remaining_capacity]
                    if not feasible_nodes:
                        break
                    # Prevent a dead loop when a constructive heuristic returns the depot too early.
                    next_node = min(
                        feasible_nodes,
                        key=lambda node: (distance_matrix[current_node, node], node),
                    )

                if next_node not in unvisited:
                    raise ValueError(f"Solver selected invalid node {next_node}")
                if demands_array[next_node] > remaining_capacity:
                    if len(route) > 1:
                        break
                    feasible_nodes = [node for node in candidate_nodes if demands_array[node] <= remaining_capacity]
                    if not feasible_nodes:
                        break
                    next_node = min(
                        feasible_nodes,
                        key=lambda node: (distance_matrix[current_node, node], node),
                    )

                route.append(next_node)
                current_node = next_node
                remaining_capacity -= int(demands_array[next_node])
                unvisited.remove(next_node)

            route.append(depot_index)
            if len(route) <= 2:
                raise ValueError("Constructive heuristic failed to serve any remaining customer")
            routes.append(route)

        return routes

    return solve_vrp
