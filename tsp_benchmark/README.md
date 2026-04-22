# TSP Benchmark

This directory contains the TSP benchmark code, benchmark data, and finalized comparison algorithms used in the public repository.

## Included Algorithms

- `StructEvo`
- `EOHU`
- `FunSearch`
- `EoH`
- `ReEvo`
- `EoHS`
- `PartEvo`

The default algorithm list is defined in `config/default_experiment.json`.

## Example Commands

Run the benchmark on instances with dimension up to 150:

```bash
python run_experiment_le150.py
```

Run the full configured benchmark:

```bash
python run_experiment.py
```

Run the StructEvo budget sweep:

```bash
python run_structevo_budget_sweep_le150.py
```

## Outputs

All results are generated locally under `outputs/` when experiments are executed. No precomputed outputs are distributed in this release.
