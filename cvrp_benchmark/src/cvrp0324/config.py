from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_FAMILIES = ("A", "B", "E", "F", "M", "P", "X")


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "path": self.path}


@dataclass(frozen=True)
class ExperimentConfig:
    label: str
    data_dir: str
    families: tuple[str, ...]
    excluded_instances: tuple[str, ...]
    algorithms: tuple[AlgorithmConfig, ...]
    num_runs: int
    processes: int
    alpha: float
    baseline_algorithm: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        algorithms = tuple(AlgorithmConfig(**item) for item in payload["algorithms"])
        families = tuple(payload.get("families") or DEFAULT_FAMILIES)
        processes = int(payload.get("processes") or max(1, min(20, os.cpu_count() or 1)))
        return cls(
            label=str(payload.get("label", "experiment")),
            data_dir=str(payload.get("data_dir", "data/cvrp")),
            families=families,
            excluded_instances=tuple(payload.get("excluded_instances") or ()),
            algorithms=algorithms,
            num_runs=int(payload.get("num_runs", 30)),
            processes=processes,
            alpha=float(payload.get("alpha", 0.05)),
            baseline_algorithm=str(payload.get("baseline_algorithm", algorithms[0].name)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "data_dir": self.data_dir,
            "families": list(self.families),
            "excluded_instances": list(self.excluded_instances),
            "algorithms": [algorithm.to_dict() for algorithm in self.algorithms],
            "num_runs": self.num_runs,
            "processes": self.processes,
            "alpha": self.alpha,
            "baseline_algorithm": self.baseline_algorithm,
        }

    def with_overrides(
        self,
        *,
        label: str | None = None,
        num_runs: int | None = None,
        processes: int | None = None,
        families: Iterable[str] | None = None,
        algorithm_names: Iterable[str] | None = None,
        baseline_algorithm: str | None = None,
    ) -> "ExperimentConfig":
        selected_algorithms = self.algorithms
        if algorithm_names is not None:
            requested = [name for name in algorithm_names if name]
            lookup = {algorithm.name: algorithm for algorithm in self.algorithms}
            missing = [name for name in requested if name not in lookup]
            if missing:
                raise ValueError(f"Unknown algorithm(s): {', '.join(missing)}")
            selected_algorithms = tuple(lookup[name] for name in requested)

        chosen_families = tuple(families) if families is not None else self.families
        chosen_baseline = baseline_algorithm or self.baseline_algorithm
        available_names = {algorithm.name for algorithm in selected_algorithms}
        if chosen_baseline not in available_names:
            raise ValueError(
                f"Baseline algorithm '{chosen_baseline}' is not in the selected algorithm list"
            )

        return ExperimentConfig(
            label=label or self.label,
            data_dir=self.data_dir,
            families=chosen_families,
            excluded_instances=self.excluded_instances,
            algorithms=selected_algorithms,
            num_runs=num_runs or self.num_runs,
            processes=processes or self.processes,
            alpha=self.alpha,
            baseline_algorithm=chosen_baseline,
        )

    def resolve_data_dir(self, root_dir: Path) -> Path:
        return resolve_path(root_dir, self.data_dir)

    def resolve_algorithms(self, root_dir: Path) -> list[tuple[str, Path]]:
        return [(algorithm.name, resolve_path(root_dir, algorithm.path)) for algorithm in self.algorithms]


def resolve_path(root_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root_dir / path
    return path.resolve()


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ExperimentConfig.from_dict(payload)
