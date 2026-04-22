from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_OPT_LINE_PATTERN = re.compile(r"^\s*([^:]+)\s*:\s*([0-9]+(?:\.[0-9]+)?)")
_SERIES_PATTERN = re.compile(r"^[A-Za-z]+")
_TSPLIB_PI = 3.141592
_EARTH_RADIUS = 6378.388


@dataclass(frozen=True)
class ProblemMetadata:
    name: str
    dimension: int
    edge_weight_type: str
    edge_weight_format: str
    has_node_coords: bool
    has_display_coords: bool

    @property
    def has_geometry(self) -> bool:
        return self.has_node_coords or self.has_display_coords


@dataclass(frozen=True)
class ParsedTSPProblem:
    name: str
    dimension: int
    edge_weight_type: str
    edge_weight_format: str
    coordinates: list[tuple[float, float]] | None
    geometry_source: str | None
    distance_matrix: np.ndarray


@dataclass(frozen=True)
class LoadedTSPInstance:
    name: str
    series: str
    tsp_path: Path
    dimension: int
    edge_weight_type: str
    edge_weight_format: str
    coordinates: list[tuple[float, float]] | None
    geometry_source: str | None
    opt_cost: float | None
    distance_matrix: np.ndarray
    evaluation_distance_matrix: np.ndarray


def collect_tsp_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.tsp"))


def build_instance_catalog(
    *,
    data_dir: Path,
    known_optima_path: Path,
    opt_tour_dir: Path,
    max_dimension: int | None = None,
    max_instances: int | None = None,
    require_geometry: bool = True,
) -> pd.DataFrame:
    known_optima = load_known_optima(known_optima_path)
    opt_tour_index = build_opt_tour_index(opt_tour_dir)

    rows: list[dict[str, object]] = []
    for tsp_path in collect_tsp_paths(data_dir):
        metadata = read_problem_metadata(tsp_path)
        key = normalize_instance_name(metadata.name or tsp_path.stem)
        opt_cost = known_optima.get(key)
        has_opt_tour = key in opt_tour_index
        eligible = (opt_cost is not None or has_opt_tour) and (metadata.has_geometry or not require_geometry)
        if max_dimension is not None and metadata.dimension > max_dimension:
            eligible = False

        instance_name = sanitize_instance_name(metadata.name or tsp_path.stem)
        rows.append(
            {
                "Instance": instance_name,
                "Series": derive_series(instance_name),
                "Dimension": metadata.dimension,
                "EdgeWeightType": metadata.edge_weight_type,
                "EdgeWeightFormat": metadata.edge_weight_format,
                "HasNodeCoords": metadata.has_node_coords,
                "HasDisplayCoords": metadata.has_display_coords,
                "HasGeometry": metadata.has_geometry,
                "OptCost": float(opt_cost) if opt_cost is not None else np.nan,
                "HasOptTour": has_opt_tour,
                "Eligible": eligible,
                "Path": str(tsp_path),
                "SizeBucket": size_bucket(metadata.dimension),
            }
        )

    catalog = pd.DataFrame(rows).sort_values(["Dimension", "Instance"], kind="stable").reset_index(drop=True)
    if max_instances is not None:
        eligible_mask = catalog["Eligible"].fillna(False)
        eligible_indices = catalog[eligible_mask].index[:max_instances]
        limited_mask = catalog.index.isin(eligible_indices)
        catalog["Eligible"] = eligible_mask & limited_mask
    return catalog


def load_instance(tsp_path: Path, known_optima_path: Path, opt_tour_dir: Path) -> LoadedTSPInstance:
    problem = parse_tsp_problem(tsp_path)
    key = normalize_instance_name(problem.name or tsp_path.stem)
    known_optima = load_known_optima(known_optima_path)
    opt_cost = known_optima.get(key)
    if opt_cost is None:
        opt_tour_path = build_opt_tour_index(opt_tour_dir).get(key)
        if opt_tour_path is not None:
            opt_tour = parse_opt_tour(opt_tour_path)
            opt_cost = calculate_tour_cost(opt_tour, problem.distance_matrix)

    solver_coordinates = problem.coordinates
    solver_distance_matrix = problem.distance_matrix
    if problem.coordinates and problem.edge_weight_type == "EUC_2D":
        # Match the original training/evaluation convention used by the generated TSP heuristics:
        # solvers operate on coordinates scaled to [0, 1]^2, while benchmark cost is still
        # measured on the original TSPLIB distance matrix.
        solver_coordinates = normalize_coordinates(problem.coordinates)
        solver_distance_matrix = build_continuous_distance_matrix(solver_coordinates)

    return LoadedTSPInstance(
        name=sanitize_instance_name(problem.name or tsp_path.stem),
        series=derive_series(sanitize_instance_name(problem.name or tsp_path.stem)),
        tsp_path=tsp_path,
        dimension=problem.dimension,
        edge_weight_type=problem.edge_weight_type,
        edge_weight_format=problem.edge_weight_format,
        coordinates=solver_coordinates,
        geometry_source=problem.geometry_source,
        opt_cost=float(opt_cost) if opt_cost is not None else None,
        distance_matrix=solver_distance_matrix,
        evaluation_distance_matrix=problem.distance_matrix,
    )


def parse_tsp_problem(tsp_path: Path) -> ParsedTSPProblem:
    metadata = read_problem_metadata(tsp_path)
    node_coords: list[tuple[float, float]] = []
    display_coords: list[tuple[float, float]] = []
    weights: list[float] = []
    section: str | None = None

    with tsp_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line == "EOF":
                break

            if line.startswith("NODE_COORD_SECTION"):
                section = "NODE_COORD_SECTION"
                continue
            if line.startswith("DISPLAY_DATA_SECTION"):
                section = "DISPLAY_DATA_SECTION"
                continue
            if line.startswith("EDGE_WEIGHT_SECTION"):
                section = "EDGE_WEIGHT_SECTION"
                continue
            if line.startswith("TOUR_SECTION"):
                section = "TOUR_SECTION"
                continue

            if section == "NODE_COORD_SECTION":
                parts = line.split()
                if len(parts) >= 3:
                    node_coords.append((float(parts[1]), float(parts[2])))
            elif section == "DISPLAY_DATA_SECTION":
                parts = line.split()
                if len(parts) >= 3:
                    display_coords.append((float(parts[1]), float(parts[2])))
            elif section == "EDGE_WEIGHT_SECTION":
                weights.extend(float(part) for part in line.split())

    coordinates: list[tuple[float, float]] | None = None
    geometry_source: str | None = None
    if node_coords:
        coordinates = node_coords
        geometry_source = "NODE_COORD_SECTION"
    elif display_coords:
        coordinates = display_coords
        geometry_source = "DISPLAY_DATA_SECTION"

    if metadata.edge_weight_type == "EXPLICIT":
        distance_matrix = build_explicit_distance_matrix(
            metadata.dimension,
            weights,
            metadata.edge_weight_format,
        )
    else:
        if not coordinates:
            raise ValueError(f"No coordinates found for non-explicit TSP file: {tsp_path}")
        distance_matrix = build_distance_matrix(coordinates, metadata.edge_weight_type)

    return ParsedTSPProblem(
        name=sanitize_instance_name(metadata.name),
        dimension=metadata.dimension,
        edge_weight_type=metadata.edge_weight_type,
        edge_weight_format=metadata.edge_weight_format,
        coordinates=coordinates,
        geometry_source=geometry_source,
        distance_matrix=distance_matrix,
    )


def read_problem_metadata(tsp_path: Path) -> ProblemMetadata:
    name = tsp_path.stem
    dimension = 0
    edge_weight_type = "EUC_2D"
    edge_weight_format = ""
    has_node_coords = False
    has_display_coords = False

    with tsp_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line == "EOF":
                break
            if line.startswith("NAME"):
                name = sanitize_instance_name(_extract_metadata_value(line) or name)
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
                has_node_coords = True
                continue
            if line.startswith("DISPLAY_DATA_SECTION"):
                has_display_coords = True
                continue

    return ProblemMetadata(
        name=name,
        dimension=dimension,
        edge_weight_type=edge_weight_type,
        edge_weight_format=edge_weight_format,
        has_node_coords=has_node_coords,
        has_display_coords=has_display_coords,
    )


def build_distance_matrix(
    coordinates: list[tuple[float, float]],
    edge_weight_type: str,
) -> np.ndarray:
    dimension = len(coordinates)
    matrix = np.zeros((dimension, dimension), dtype=float)
    for row in range(dimension):
        for col in range(row + 1, dimension):
            value = distance_between(coordinates[row], coordinates[col], edge_weight_type)
            matrix[row, col] = value
            matrix[col, row] = value
    return matrix


def build_continuous_distance_matrix(coordinates: list[tuple[float, float]]) -> np.ndarray:
    dimension = len(coordinates)
    matrix = np.zeros((dimension, dimension), dtype=float)
    for row in range(dimension):
        for col in range(row + 1, dimension):
            value = float(math.dist(coordinates[row], coordinates[col]))
            matrix[row, col] = value
            matrix[col, row] = value
    return matrix


def normalize_coordinates(coordinates: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not coordinates:
        return []

    xs = [coord[0] for coord in coordinates]
    ys = [coord[1] for coord in coordinates]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    scale = max(x_max - x_min, y_max - y_min)
    if scale <= 0:
        return [(0.0, 0.0) for _ in coordinates]

    return [((x - x_min) / scale, (y - y_min) / scale) for x, y in coordinates]


def distance_between(
    coord_a: tuple[float, float],
    coord_b: tuple[float, float],
    edge_weight_type: str,
) -> float:
    if edge_weight_type == "ATT":
        xd = coord_a[0] - coord_b[0]
        yd = coord_a[1] - coord_b[1]
        rij = math.sqrt((xd * xd + yd * yd) / 10.0)
        tij = int(rij + 0.5)
        return float(tij + 1 if tij < rij else tij)

    if edge_weight_type == "CEIL_2D":
        return float(math.ceil(math.dist(coord_a, coord_b)))

    if edge_weight_type == "GEO":
        lat_a = _geo_to_radians(coord_a[0])
        lon_a = _geo_to_radians(coord_a[1])
        lat_b = _geo_to_radians(coord_b[0])
        lon_b = _geo_to_radians(coord_b[1])
        q1 = math.cos(lon_a - lon_b)
        q2 = math.cos(lat_a - lat_b)
        q3 = math.cos(lat_a + lat_b)
        value = _EARTH_RADIUS * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1.0
        return float(int(value))

    return float(int(math.dist(coord_a, coord_b) + 0.5))


def build_explicit_distance_matrix(
    dimension: int,
    weights: list[float],
    edge_weight_format: str,
) -> np.ndarray:
    matrix = np.zeros((dimension, dimension), dtype=float)
    index = 0

    if edge_weight_format == "FULL_MATRIX":
        if len(weights) != dimension * dimension:
            raise ValueError(
                f"Unexpected number of weights for FULL_MATRIX: {len(weights)} vs {dimension * dimension}"
            )
        return np.array(weights, dtype=float).reshape(dimension, dimension)

    if edge_weight_format == "UPPER_ROW":
        for row in range(dimension - 1):
            for col in range(row + 1, dimension):
                matrix[row, col] = weights[index]
                matrix[col, row] = weights[index]
                index += 1
    elif edge_weight_format == "LOWER_ROW":
        for row in range(1, dimension):
            for col in range(row):
                matrix[row, col] = weights[index]
                matrix[col, row] = weights[index]
                index += 1
    elif edge_weight_format == "UPPER_DIAG_ROW":
        for row in range(dimension):
            for col in range(row, dimension):
                matrix[row, col] = weights[index]
                matrix[col, row] = weights[index]
                index += 1
    elif edge_weight_format == "LOWER_DIAG_ROW":
        for row in range(dimension):
            for col in range(row + 1):
                matrix[row, col] = weights[index]
                matrix[col, row] = weights[index]
                index += 1
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}")

    if index != len(weights):
        raise ValueError(f"Unexpected number of explicit weights: consumed {index}, total {len(weights)}")
    return matrix


def validate_tour(tour: object, dimension: int) -> tuple[bool, str, list[int] | None]:
    if tour is None:
        return False, "Solver returned None", None
    if not isinstance(tour, (list, tuple, np.ndarray)):
        return False, "Solver must return a sequence of nodes", None

    normalized = [int(node) for node in list(tour)]
    if len(normalized) == dimension + 1 and normalized[0] == normalized[-1]:
        normalized = normalized[:-1]

    if len(normalized) != dimension:
        return False, f"Tour length mismatch ({len(normalized)} != {dimension})", None

    if any(node < 0 or node >= dimension for node in normalized):
        return False, "Tour contains out-of-range nodes", None

    if len(set(normalized)) != dimension:
        return False, "Tour contains duplicate nodes", None

    return True, "OK", normalized


def calculate_tour_cost(tour: list[int], distance_matrix: np.ndarray) -> float:
    total_cost = 0.0
    for start, end in zip(tour[:-1], tour[1:]):
        total_cost += float(distance_matrix[start, end])
    total_cost += float(distance_matrix[tour[-1], tour[0]])
    return total_cost


def parse_opt_tour(opt_tour_path: Path) -> list[int]:
    section = None
    nodes: list[int] = []
    with opt_tour_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("TOUR_SECTION"):
                section = "TOUR_SECTION"
                continue
            if line == "EOF":
                break
            if section == "TOUR_SECTION":
                for part in line.split():
                    value = int(part)
                    if value == -1:
                        return [node - 1 for node in nodes]
                    nodes.append(value)
    return [node - 1 for node in nodes]


@lru_cache(maxsize=8)
def load_known_optima(known_optima_path: Path) -> dict[str, float]:
    optima: dict[str, float] = {}
    with known_optima_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            match = _OPT_LINE_PATTERN.match(raw_line.strip())
            if not match:
                continue
            name = normalize_instance_name(match.group(1))
            optima.setdefault(name, float(match.group(2)))
    return optima


@lru_cache(maxsize=8)
def build_opt_tour_index(opt_tour_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not opt_tour_dir.exists():
        return index
    for opt_tour_path in opt_tour_dir.glob("*.opt.tour"):
        base_name = opt_tour_path.name.replace(".opt.tour", "")
        index[normalize_instance_name(base_name)] = opt_tour_path
    return index


def normalize_instance_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def derive_series(name: str) -> str:
    match = _SERIES_PATTERN.match(name)
    return match.group(0) if match else name


def sanitize_instance_name(name: str) -> str:
    return name[:-4] if name.lower().endswith(".tsp") else name


def size_bucket(dimension: int) -> str:
    if dimension <= 50:
        return "<=50"
    if dimension <= 100:
        return "51-100"
    if dimension <= 150:
        return "101-150"
    if dimension <= 300:
        return "151-300"
    if dimension <= 1000:
        return "301-1000"
    return ">1000"


def _extract_metadata_value(line: str) -> str:
    if ":" in line:
        return line.split(":", 1)[1].strip()
    return line.split()[-1].strip()


def _geo_to_radians(value: float) -> float:
    degrees = int(value)
    minutes = value - degrees
    return _TSPLIB_PI * (degrees + 5.0 * minutes / 3.0) / 180.0
