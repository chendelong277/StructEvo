# EoH-U and Benchmark Repository

This package contains the public source release without the StructEvo framework source code. It includes the EoH-U source code and the TSP/CVRP benchmark code with the finalized comparison algorithms used in evaluation, including the finalized StructEvo benchmark algorithms.

![StructEvo framework](docs/assets/structevo-framework.png)

## Contents

```text
eohu_public/
├── eohu/                  # EoH-U source
├── tsp_benchmark/         # TSP benchmark and comparison algorithms
├── cvrp_benchmark/        # CVRP benchmark and comparison algorithms
├── docs/assets/           # Documentation figures
├── requirements.txt
└── README.md
```

## Installation

Use Python 3.11 or later.

```bash
pip install -r requirements.txt
```

## Configuration

API settings are stored in:

- `eohu/config/tsp_config.py`
- `eohu/config/cvrp_config.py`

The default model is `deepseek-v3`. The default API key and base URL are placeholders and must be replaced before execution.

## Running EoH-U

```bash
python eohu/main.py --problem tsp
python eohu/main.py --problem cvrp
```

## Running Benchmarks

TSP benchmark on instances with dimension up to 150:

```bash
python tsp_benchmark/run_experiment_le150.py
```

CVRP benchmark on instances with dimension up to 150:

```bash
python cvrp_benchmark/run_experiment_le150.py
```

TSP StructEvo budget sweep over finalized benchmark algorithms:

```bash
python tsp_benchmark/run_structevo_budget_sweep_le150.py
```

## Notes

- This package does not include the StructEvo framework source code.
- The benchmark directories retain the finalized StructEvo and EoH-U algorithms used for comparison experiments.
- No experiment outputs are included in this release. Result directories are generated only after running the code locally.
- Benchmark outputs are written to newly created `outputs/` directories and are excluded by `.gitignore`.
