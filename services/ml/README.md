# ML Service

Python service for research workflows around:

- synthetic LMS-like data generation,
- weekly student twin snapshot construction,
- schema-aware validation,
- feature engineering and versioned experiments,
- explainability and public-benchmark artifact generation.

## Dependency Management

This service uses **uv** for fast, reproducible Python dependency management across research iterations.

## Windows Quick Start

From the repository root, create and activate a virtual environment.

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then install dependencies with `pip`:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ./services/ml
```

If you prefer a non-editable install, use:

```powershell
python -m pip install ./services/ml
```

If you prefer a plain requirements file, use:

```powershell
python -m pip install -r services/ml/requirements.txt
```

## Dataset Generation

The current entrypoint builds the synthetic dataset pipeline aligned with `schema_v1.2`.

## Benchmark Configs

The repository now keeps explicit benchmark configs instead of relying on one mutable file:

- `configs/generator_v1_2_baseline.yaml`: benchmark baseline using the v1.2 trajectory tuning
- `configs/generator_v1_3_refined.yaml`: recommended refined benchmark using the v1.3 realism tuning
- `configs/generator_v1.yaml`: convenience alias for the current recommended refined setup

Run from `services/ml`:

With `uv`:

```powershell
uv run python -m src.main --config configs/generator_v1.yaml
```

With the activated `venv` and plain `pip` install:

```powershell
python -m src.main --config configs/generator_v1.yaml
```

Recommended benchmark runs:

```powershell
python -m src.main --config configs/generator_v1_2_baseline.yaml --output-root C:\path\to\dataset_v1_2
python -m src.main --config configs/generator_v1_3_refined.yaml --output-root C:\path\to\dataset_v1_3
```

Optional flags:

- `--seed 123`
- `--output-root ../../tmp/dataset_run`
- `--skip-parquet`

Outputs are written to:

- `data/raw/*.csv`
- `data/processed/student_twin_snapshots.csv`
- `data/processed/student_twin_snapshots.parquet` unless skipped
- `data/artifacts/reports/realism_metrics.json`
- `data/artifacts/reports/realism_report.md`

## What the Pipeline Does

1. Loads the generator config and resolves output paths.
2. Generates raw LMS-like tables for the one-course, 10-week prototype.
3. Builds `student_twin_snapshots` with `1 row = 1 student x 1 week`.
4. Validates generated tables against `packages/contracts/schema_versions/schema_v1.2.yaml`.
5. Runs a realism audit and writes JSON + Markdown reports.
6. Writes outputs and prints a concise summary.

## Comparison Reports

To compare two generated dataset runs:

```powershell
python -m src.compare --left C:\path\to\dataset_v1_2 --right C:\path\to\dataset_v1_3
```

This writes:

- `comparison_summary.json`
- `comparison_report.md`
- `key_metrics.csv`

under `data/artifacts/reports/comparisons/` by default.

## Baseline Experiments

The first experimental phase is preserved as `exp_001_baseline`. It compares
three feature sets across simple baseline models to see whether the Digital Twin
representation improves prediction over plain LMS-style features. The
supervised targets are:

- regression: `final_results.final_grade`
- classification: `final_results.passed`

The teacher-facing `risk_level` heuristic is intentionally **not** used as a
supervised target — it is already derived from the same weekly snapshot
features and would short-circuit the comparison.

### Feature sets

| Name | Intent | Highlights |
| --- | --- | --- |
| `A_simple` | Minimal academic baseline | assignment / quiz averages + attendance rate |
| `B_lms` | Stronger non-twin LMS baseline | adds activity, time on platform, discipline + on-time rate |
| `C_twin` | Full Digital Twin representation | adds trend features, mastery proxies, composite indices |

Membership is declared in `src/experiments/featuresets.py`; forbidden columns
(identifiers, the heuristic risk fields, the snapshot-level
`predicted_final_grade`, end-of-course outcomes, generation-only fields) are
guarded against accidental leakage.

### Splits

- Default: leakage-safe **student-grouped** split (`test_size = 0.25`).
- Secondary: **temporal-forward** split that holds out late weeks for held-out
  students. This combines week and student isolation, which is stricter than
  pure week-only splitting.
- Row-wise random split is rejected because each student contributes many
  weekly rows.

### Run the baselines

From `services/ml`:

```powershell
python -m src.experiments.run_baselines --config configs/experiments/exp_001_baseline.yaml
```

The runner:

1. loads the refined v1.3 dataset artifacts,
2. joins outcomes from `final_results`,
3. produces an experiment-oriented EDA report,
4. trains three baseline classifiers and three baseline regressors per
   feature set, per split,
5. writes CSV / JSON result tables and a markdown summary.

Outputs:

- preserved legacy outputs under `data/artifacts/experiments/baselines/`
- versioned copies and metadata under
  `data/artifacts/experiments/exp_001_baseline/`

To inspect the declared feature sets without running the experiments:

```powershell
python -m src.experiments.run_baselines --list-feature-sets
```

## Versioned Experiments

The completed experiment line is frozen for dissertation packaging:

- `exp_001_baseline`: baseline feature-set comparison.
- `exp_002_twin_ablation`: Twin subgroup ablation.
- `exp_003_mastery_validation`: mastery-block validation.
- `exp_004_xai_on_lean_twin`: XAI on the lean Twin candidate.
- `exp_005_public_benchmark_oulad`: OULAD public-benchmark transfer stress
  test.

The synthetic experiments use schema `v1.2`, keep `final_grade` as the primary
target, and report `passed` only as secondary context. `risk_level` is
intentionally excluded from supervised training. The OULAD benchmark uses the
external adapter schema `external_oulad_adapter_v1` and the derived
`final_weighted_score` target.

Run from `services/ml`:

```powershell
python -m src.experiments.run_ablation --config configs/experiments/exp_002_twin_ablation.yaml
python -m src.experiments.run_mastery_validation --config configs/experiments/exp_003_mastery_validation.yaml
python -m src.experiments.run_xai_on_lean_twin --config configs/experiments/exp_004_xai_on_lean_twin.yaml
python -m src.experiments.run_public_benchmark_oulad --config configs/experiments/exp_005_public_benchmark_oulad.yaml
```

Each runner writes versioned outputs under
`data/artifacts/experiments/<experiment_id>/`, updates the corresponding
human-readable experiment documentation, and preserves prior experiment
artifacts by experiment ID. Do not rerun the full experiment stack during
dissertation packaging unless a specific verification requires it.

The runner family writes:

- `experiment_metadata.json`
- JSON / CSV / Markdown result tables
- experiment-specific recommendation or diagnostic artifacts
- `docs/experiments/<experiment_id>.md`
- an updated row in `docs/experiments/registry.md`

Experiment conventions are documented in `docs/experiments/README.md`.

## Testing

Run from `services/ml`:

```powershell
python -m pytest
```

If you want the same mode used during local verification in this repository:

```powershell
python -m pytest tests -v -p no:cacheprovider
```
