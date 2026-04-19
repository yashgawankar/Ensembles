# Bagging vs Boosting · Bias-Variance Analysis Pipeline

A fully modular, reusable pipeline for decomposing prediction error into **bias²** and **variance** across ensemble methods (Random Forest, XGBoost, Bagged-XGB) and hybrid mixing strategies, with both synthetic and real-world dataset support.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Full synthetic run (~1-2 hours, 50 bootstrap iterations)
python main.py

# Fast smoke test (~2 minutes)
python main.py --quick --n-bootstrap 5

# California Housing dataset
python main.py --dataset california

# Only run specific experiments
python main.py --experiments ensemble_size depth

# Resume a previous run (already-computed configs are skipped automatically)
python main.py --results-dir ./results
```

---

## Project Structure

```
bias_variance_pipeline/
├── config.py                    ← Global constants, baseline HPs, sweep values, colours
├── pipeline_types.py            ← Shared dataclasses (BiasVarianceResult, ConfigResult, …)
├── main.py                      ← Top-level execution script (all steps, all flags)
├── requirements.txt
│
├── data/
│   ├── generators.py            ← DataGenerator ABC + Synthetic / CaliforniaHousing / CSV
│   └── preprocessor.py          ← DataPreprocessor + DatasetVariantManager
│
├── models/
│   └── model_builder.py         ← HyperparameterRegistry + ModelBuilder factory
│
├── experiments/
│   ├── bootstrap_evaluator.py   ← BootstrapEvaluator (single + paired)
│   ├── orchestrator.py          ← ExperimentOrchestrator (E1–E4)
│   ├── hybrid_engine.py         ← HybridComputationEngine (E5–E6, convex + probabilistic)
│   └── theory_validator.py      ← TheoryValidator (analytical λ* + comparison)
│
├── analysis/
│   ├── results_store.py         ← ResultsStore (in-memory + disk, resumable)
│   └── metrics.py               ← MetricsComputer + TableBuilder (H1–H7 verdicts)
│
└── visualization/
    ├── trajectory_plotter.py    ← TrajectoryPlotter (Plots 1–4)
    └── hybrid_plotter.py        ← HybridPlotter (Plots 5–6)
```

---

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `synthetic` | `synthetic` · `california` · `csv` |
| `--csv-path` | — | Path to CSV file (required if `--dataset csv`) |
| `--target-col` | — | Target column name (required if `--dataset csv`) |
| `--results-dir` | `./results` | Where to save / resume from |
| `--plots-dir` | `./plots` | Where to save plots |
| `--n-bootstrap` | `50` | Number of bootstrap iterations |
| `--experiments` | all four | Subset: `ensemble_size` `depth` `noise` `train_size` |
| `--quick` | off | Reduces sweep values to 2 per experiment for fast testing |
| `--no-plots` | off | Skip plot generation |
| `--verbose` | off | Per-bootstrap-iteration logging |

---

## Experiments

| ID | What varies | Values | Models |
|----|------------|--------|--------|
| E1 | Ensemble size (`n_estimators`) | 5, 10, 25, 50, 100, 200 | RF · XGB · Bagged-XGB |
| E2 | Base learner depth (`max_depth`) | 2, 4, 6, 8, 10, 15, 20 (+ None for RF) | RF · XGB · Bagged-XGB |
| E3 | Noise level (σ) | 5, 15, 30, 50, 75 | RF · XGB · Bagged-XGB |
| E4 | Training set size | 500, 1k, 2k, 4k, 8k | RF · XGB · Bagged-XGB |
| E5 | Convex hybrid λ sweep | 0.0 → 1.0 by 0.1 | RF + XGB blend |
| E6 | Probabilistic hybrid λ sweep | 0.0 → 1.0 by 0.1 | RF + XGB stochastic |

---

## Outputs

### Results directory (`./results/`)
```
results/
├── experiments/
│   ├── ensemble_size.json       ← scalar metrics for all configs
│   ├── depth.json
│   ├── noise.json
│   └── train_size.json
├── hybrids/
│   ├── convex_baseline.json
│   └── probabilistic_baseline.json
├── predictions/
│   └── rf__ensemble_size__100.npy   ← (M × n_test) bootstrap prediction matrices
├── metadata/
│   └── metadata.json
├── all_results.csv              ← flat DataFrame of all scalar metrics
├── theory_validation.csv        ← λ* theory vs empirical comparison table
└── hypothesis_verdicts.csv      ← H1–H7 programmatic verdicts
```

### Plots directory (`./plots/`)
| File | Description |
|------|-------------|
| `plot_ensemble_size.png` | E1 trajectory in Bias²-Variance space |
| `plot_depth.png` | E2 trajectory |
| `plot_noise.png` | E3 trajectory |
| `plot_train_size.png` | E4 trajectory |
| `plot_hybrid_bv_space.png` | Convex vs probabilistic curves (Plot 5) |
| `plot_mse_vs_lambda.png` | MSE vs λ with theoretical λ* (Plot 6) |

---

## Switching Datasets

### California Housing (drop-in replacement)
```python
# In your own script:
from data.generators import CaliforniaHousingDataGenerator
from data.preprocessor import DataPreprocessor, DatasetVariantManager

gen  = CaliforniaHousingDataGenerator()
prep = DataPreprocessor(gen, test_size=2000, random_seed=42)
prep.initialise()
```
Or via CLI:
```bash
python main.py --dataset california
```
> Note: noise variants are silently skipped for real datasets (noise is inherent in the data).

### Custom CSV
```bash
python main.py --dataset csv --csv-path ./my_data.csv --target-col price
```

---

## Resumability

The pipeline writes results to disk after every model configuration. If a run is interrupted, restart with the same `--results-dir` and already-computed configs will be loaded from disk instead of retrained:

```bash
# First run (interrupted after E1)
python main.py --results-dir ./results

# Resume — E1 is skipped, E2–E4 continue from where they left off
python main.py --results-dir ./results
```

---

## Extending the Pipeline

### Add a new model (e.g. LightGBM)
```python
# In models/model_builder.py ModelBuilder.build():
elif model_type == 'lgb':
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**config)

# In config.py BASELINE_PARAMS:
'lgb': {'n_estimators': 100, 'max_depth': 7, 'random_state': 42}

# In models/model_builder.py HyperparameterRegistry._build_default_variations():
registry._add_raw('ensemble_size', 'lgb', 'n_estimators', val)
```

### Add a new experiment (e.g. feature subset size)
```python
# In config.py EXPERIMENT_VARIATIONS:
'feature_count': {
    'label': 'Feature Count',
    'values': [3, 6, 9, 12, 15],
    ...
}
# Then create DatasetVariantManager variants with subsetted X columns.
```

### Change sweep values without editing config.py
```python
registry = HyperparameterRegistry()
registry._variations['ensemble_size'].clear()
for val in [10, 100, 500, 1000]:
    registry._add_raw('ensemble_size', 'rf',  'n_estimators', val)
    registry._add_raw('ensemble_size', 'xgb', 'n_estimators', val)
    registry._add_raw('ensemble_size', 'bagged_xgb', 'outer_n_estimators', val)
```

---

## Mathematical Definitions

### Bias-Variance Decomposition (Bootstrap)
```
mean_pred[j]  = (1/M) Σᵢ ŷᵢ[j]         (average over M bootstrap runs)
bias²         = (1/n) Σⱼ (mean_pred[j] - y[j])²
variance      = (1/n) Σⱼ Var_i(ŷᵢ[j])
MSE           = bias² + variance
```

### Convex Hybrid
```
ŷ_λ = λ · ŷ_XGB + (1-λ) · ŷ_RF

Var(ŷ_λ) = λ²·Var_XGB + (1-λ)²·Var_RF + 2λ(1-λ)·Cov(RF, XGB)

λ* = (Var_RF - Cov - B_RF·ΔB) / (ΔB² + Var_XGB + Var_RF - 2·Cov)
     where ΔB = B_XGB - B_RF
```

### Probabilistic Hybrid
```
P(use XGB) = λ,  P(use RF) = 1-λ   (per prediction, per bootstrap run)

Var(ŷ_λ) = λ·Var_XGB + (1-λ)·Var_RF + λ(1-λ)·(μ_XGB - μ_RF)²

λ* = (Var_RF - Var_XGB - D² - 2·B_RF·ΔB) / (2·ΔB² - 2·D²)
     where D² = (μ_XGB - μ_RF)² averaged over test points
```

---

## Hypotheses Being Tested

| ID | Hypothesis |
|----|-----------|
| H1 | Bagging trajectories move ↓ (variance reduction); boosting moves ← then ↑ |
| H2 | XGBoost is more sensitive to base learner depth than RF |
| H3 | XGBoost benefits more from larger training sets than RF |
| H4 | XGBoost variance degrades more under high noise than RF |
| H5 | Optimal λ* achieves lower MSE than either parent model alone |
| H6 | Theoretical λ* matches empirical λ* within 0.1 |
| H7 | Convex hybrid achieves lower minimum MSE than probabilistic hybrid |

---

## Design Principles

- **Immutable test set**: same 2,000 test points throughout all experiments (fixed seed split)
- **Paired bootstrap**: RF and XGB always trained on the same bootstrap sample for valid covariance estimation
- **Resumability**: all results written to disk after each config; restarts load from cache
- **Data abstraction**: swap datasets by changing one class, rest of pipeline unchanged
- **Prediction storage**: full (M × n_test) matrices kept on disk; hybrids reuse without retraining
- **Deterministic seeds**: global seed + per-iteration offsets guarantee exact reproducibility
