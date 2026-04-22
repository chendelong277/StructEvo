from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_COST_PATTERN = re.compile(r"Cost\s*:?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_FILENAME_DIMENSION_PATTERN = re.compile(r"(?:^|-)n(\d+)(?:-|$)", re.IGNORECASE)


@dataclass(frozen=True)
class LoadedInstance:
    family: str
    name: str
    vrp_path: Path
    opt_cost: float | None
    demands: list[int]
    capacity: int
    depot_index: int
    distance_matrix: np.ndarray
    constructive_distance_matrix: np.ndarray
    evaluation_distance_matrix: np.ndarray


@dataclass(frozen=True)
class ParsedVRPProblem:
    coordinates: list[tuple[float, float]] | None
    demands: list[int]
    capacity: int
    depot_index: int
    edge_weight_type: str
    distance_matrix: np.ndarray


def collect_vrp_paths(data_dir: Path, families: Iterable[str]) -> list[Path]:
    vrp_paths: list[Path] = []
    for family in families:
        family_dir = data_dir / family
        if not family_dir.exists():
            continue
        vrp_paths.extend(sorted(family_dir.glob("*.vrp")))
    return vrp_paths


def load_instance(vrp_path: Path) -> LoadedInstance:
    problem = parse_vrp_problem(vrp_path)
    constructive_distance_matrix = problem.distance_matrix
    if problem.coordinates and problem.edge_weight_type == "EUC_2D":
        # Match the normalized CVRP training distribution used during algorithm generation:
        # constructive heuristics operate on [0, 1]^2 coordinates and the corresponding
        # continuous Euclidean distance matrix, while benchmark cost is still measured on
        # the original CVRPLib integer matrix.
        normalized_coordinates = normalize_coordinates(problem.coordinates)
        constructive_distance_matrix = build_continuous_distance_matrix(normalized_coordinates)

    opt_cost = parse_solution_cost(vrp_path.with_suffix(".sol"))
    return LoadedInstance(
        family=vrp_path.parent.name,
        name=vrp_path.stem,
        vrp_path=vrp_path,
        opt_cost=opt_cost,
        demands=problem.demands,
        capacity=problem.capacity,
        depot_index=problem.depot_index,
        distance_matrix=problem.distance_matrix,
        constructive_distance_matrix=constructive_distance_matrix,
        evaluation_distance_matrix=problem.distance_matrix,
    )


def read_instance_dimension(vrp_path: Path) -> int:
    with vrp_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("DIMENSION"):
                _, _, value = line.partition(":")
                value = value.strip()
                if value:
                    return int(value)
                parts = line.split()
                return int(parts[-1])

    match = _FILENAME_DIMENSION_PATTERN.search(vrp_path.stem)
    if match:
        return int(match.group(1))

    coordinates, _, _, _ = parse_vrp_file(vrp_path)
    return len(coordinates)


def filter_vrp_paths_by_max_dimension(vrp_paths: Iterable[Path], max_dimension: int) -> list[Path]:
    selected_paths: list[Path] = []
    for vrp_path in vrp_paths:
        if read_instance_dimension(vrp_path) <= max_dimension:
            selected_paths.append(vrp_path)
    return selected_paths


def exclude_vrp_paths_by_stem(vrp_paths: Iterable[Path], excluded_instances: Iterable[str]) -> list[Path]:
    excluded = {name.strip() for name in excluded_instances if name and name.strip()}
    if not excluded:
        return list(vrp_paths)
    return [vrp_path for vrp_path in vrp_paths if vrp_path.stem not in excluded]


def parse_vrp_problem(vrp_path: Path) -> ParsedVRPProblem:
    coordinates: list[tuple[float, float]] = []
    demands: list[int] = []
    capacity = 0
    depot_index = 0
    dimension = 0
    edge_weight_type = ""
    edge_weight_format = ""
    explicit_weights: list[int] = []
    section: str | None = None

    with vrp_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line == "EOF":
                continue

            if line.startswith("CAPACITY"):
                capacity = int(_extract_metadata_value(line))
                continue
            if line.startswith("DIMENSION"):
                dimension = int(_extract_metadata_value(line))
                continue
            if line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = _extract_metadata_value(line).upper()
                continue
            if line.startswith("EDGE_WEIGHT_FORMAT"):
                edge_weight_format = _extract_metadata_value(line).upper()
                continue

            if line.startswith("NODE_COORD_SECTION"):
                section = "COORD"
                continue
            if line.startswith("DEMAND_SECTION"):
                section = "DEMAND"
                continue
            if line.startswith("DEPOT_SECTION"):
                section = "DEPOT"
                continue
            if line.startswith("EDGE_WEIGHT_SECTION"):
                section = "EDGE_WEIGHT"
                continue

            parts = line.split()
            if section == "COORD" and len(parts) >= 3:
                coordinates.append((float(parts[1]), float(parts[2])))
            elif section == "DEMAND" and len(parts) >= 2:
                demands.append(int(parts[1]))
            elif section == "DEPOT" and parts:
                depot_id = int(parts[0])
                if depot_id != -1:
                    depot_index = depot_id - 1
            elif section == "EDGE_WEIGHT":
                explicit_weights.extend(int(part) for part in parts)

    if edge_weight_type == "EXPLICIT":
        if dimension <= 0:
            raise ValueError(f"Missing DIMENSION in explicit VRP file: {vrp_path}")
        distance_matrix = build_explicit_distance_matrix(dimension, explicit_weights, edge_weight_format)
    else:
        distance_matrix = compute_distance_matrix(coordinates)

    return ParsedVRPProblem(
        coordinates=coordinates or None,
        demands=demands,
        capacity=capacity,
        depot_index=depot_index,
        edge_weight_type=edge_weight_type,
        distance_matrix=distance_matrix,
    )


def parse_vrp_file(vrp_path: Path) -> tuple[list[tuple[float, float]], list[int], int, int]:
    coordinates: list[tuple[float, float]] = []
    demands: list[int] = []
    capacity = 0
    depot_index = 0
    section: str | None = None

    with vrp_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line == "EOF":
                continue

            if line.startswith("CAPACITY"):
                _, _, value = line.partition(":")
                capacity = int(value.strip())
                continue

            if line.startswith("NODE_COORD_SECTION"):
                section = "COORD"
                continue
            if line.startswith("DEMAND_SECTION"):
                section = "DEMAND"
                continue
            if line.startswith("DEPOT_SECTION"):
                section = "DEPOT"
                continue

            parts = line.split()
            if section == "COORD" and len(parts) >= 3:
                coordinates.append((float(parts[1]), float(parts[2])))
            elif section == "DEMAND" and len(parts) >= 2:
                demands.append(int(parts[1]))
            elif section == "DEPOT" and parts:
                depot_id = int(parts[0])
                if depot_id != -1:
                    depot_index = depot_id - 1

    return coordinates, demands, capacity, depot_index


def build_explicit_distance_matrix(
    dimension: int,
    weights: list[int],
    edge_weight_format: str,
) -> np.ndarray:
    matrix = np.zeros((dimension, dimension), dtype=int)
    index = 0

    if edge_weight_format == "LOWER_ROW":
        for row in range(1, dimension):
            for col in range(row):
                value = weights[index]
                matrix[row, col] = value
                matrix[col, row] = value
                index += 1
    elif edge_weight_format == "LOWER_DIAG_ROW":
        for row in range(dimension):
            for col in range(row + 1):
                value = weights[index]
                matrix[row, col] = value
                matrix[col, row] = value
                index += 1
    elif edge_weight_format == "UPPER_ROW":
        for row in range(dimension - 1):
            for col in range(row + 1, dimension):
                value = weights[index]
                matrix[row, col] = value
                matrix[col, row] = value
                index += 1
    elif edge_weight_format == "UPPER_DIAG_ROW":
        for row in range(dimension):
            for col in range(row, dimension):
                value = weights[index]
                matrix[row, col] = value
                matrix[col, row] = value
                index += 1
    elif edge_weight_format == "FULL_MATRIX":
        matrix = np.array(weights, dtype=int).reshape(dimension, dimension)
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}")

    if index and index != len(weights):
        raise ValueError(
            f"Unexpected number of explicit edge weights: consumed {index}, total {len(weights)}"
        )
    return matrix


def compute_distance_matrix(coordinates: list[tuple[float, float]]) -> np.ndarray:
    count = len(coordinates)
    distance_matrix = np.zeros((count, count), dtype=int)
    for i in range(count):
        x1, y1 = coordinates[i]
        for j in range(count):
            if i == j:
                continue
            x2, y2 = coordinates[j]
            distance_matrix[i, j] = int(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) + 0.5)
    return distance_matrix


def build_continuous_distance_matrix(coordinates: list[tuple[float, float]]) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def normalize_coordinates(coordinates: list[tuple[float, float]]) -> list[tuple[float, float]]:
    coords = np.asarray(coordinates, dtype=float)
    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    scale = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]))
    if scale <= 0.0:
        scale = 1.0
    normalized = (coords - min_xy) / scale
    return [(float(x), float(y)) for x, y in normalized]


def parse_solution_cost(solution_path: Path) -> float | None:
    if not solution_path.exists():
        return None

    with solution_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            match = _COST_PATTERN.search(raw_line.strip())
            if match:
                return float(match.group(1))
    return None


def validate_routes(
    routes: object,
    demands: list[int],
    capacity: int,
    depot_index: int = 0,
) -> tuple[bool, str]:
    if routes is None:
        return False, "Solver returned None"
    if not isinstance(routes, (list, tuple)):
        return False, "Solver must return a list of routes"

    visited = [0] * len(demands)
    for route_index, route in enumerate(routes):
        if not isinstance(route, (list, tuple)) or len(route) < 2:
            return False, f"Route {route_index} is malformed"
        if route[0] != depot_index or route[-1] != depot_index:
            return False, f"Route {route_index} does not start/end at the depot"

        load = 0
        for node in route[1:-1]:
            if not isinstance(node, (int, np.integer)):
                return False, f"Route {route_index} contains a non-integer node"
            if node < 0 or node >= len(demands):
                return False, f"Route {route_index} contains an out-of-range node"
            if node == depot_index:
                return False, f"Route {route_index} contains the depot inside the route"
            load += demands[int(node)]
            visited[int(node)] += 1

        if load > capacity:
            return False, f"Route {route_index} exceeds capacity ({load} > {capacity})"

    missing_nodes = [index for index, count in enumerate(visited) if index != depot_index and count == 0]
    if missing_nodes:
        return False, f"Missing customer(s): {missing_nodes[:5]}"

    repeated_nodes = [index for index, count in enumerate(visited) if count > 1]
    if repeated_nodes:
        return False, f"Repeated customer(s): {repeated_nodes[:5]}"

    return True, "OK"


def calculate_total_distance(routes: list[list[int]], distance_matrix: np.ndarray) -> int:
    total_distance = 0
    for route in routes:
        for start, end in zip(route[:-1], route[1:]):
            total_distance += int(distance_matrix[start, end])
    return total_distance


def _extract_metadata_value(line: str) -> str:
    if ":" in line:
        return line.split(":", 1)[1].strip()
    return line.split()[-1].strip()
