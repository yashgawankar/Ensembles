"""
main.py — Top-level pipeline execution script.

Usage
─────
  # Full synthetic run (default)
  python main.py

  # California Housing dataset
  python main.py --dataset california

  # Custom CSV
  python main.py --dataset csv --csv-path ./data.csv --target-col price

  # Resume a previous run (skips already-computed configs)
  python main.py --results-dir ./results

  # Only run specific experiments
  python main.py --experiments ensemble_size depth

  # Quick smoke-test with fewer bootstrap runs
  python main.py --n-bootstrap 5 --quick

Flags
─────
  --dataset       : synthetic | california | csv  (default: synthetic)
  --csv-path      : path to CSV (required if --dataset csv)
  --target-col    : target column name (required if --dataset csv)
  --results-dir   : where to save / resume from (default: ./results)
  --n-bootstrap   : bootstrap iterations (default: 50)
  --experiments   : subset of experiments to run (default: all)
  --quick         : reduce sweep values for fast testing
  --no-plots      : skip plot generation
  --verbose       : detailed logging
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional

# Ensure the project root is on sys.path regardless of where python is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")

import numpy as np

# ── Pipeline imports ──────────────────────────────────────────────────────────
from config import (
    GLOBAL_SEED,
    DATASET_CONFIG,
    BOOTSTRAP_CONFIG,
    BASELINE_PARAMS,
    EXPERIMENT_VARIATIONS,
    LAMBDA_VALUES,
    THEORY_CONDITIONS,
    MODEL_LABELS,
)
from data.generators import (
    SyntheticDataGenerator,
    CaliforniaHousingDataGenerator,
    CustomCSVDataGenerator,
)
from data.preprocessor import DataPreprocessor, DatasetVariantManager
from models.model_builder import HyperparameterRegistry, ModelBuilder
from experiments.bootstrap_evaluator import BootstrapEvaluator
from experiments.orchestrator import ExperimentOrchestrator
from experiments.hybrid_engine import HybridComputationEngine
from experiments.theory_validator import TheoryValidator
from experiments.hyperparameter_tuner import HyperparameterTuner
from analysis.results_store import ResultsStore
from analysis.metrics import MetricsComputer, TableBuilder
from visualization.trajectory_plotter import TrajectoryPlotter
from visualization.hybrid_plotter import HybridPlotter
from pipeline_types import ConfigResult, ExperimentResult, TheoryValidationRow


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bagging vs Boosting Bias-Variance Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset",      default="synthetic",
                        choices=["synthetic", "california", "csv"])
    parser.add_argument("--csv-path",     default=None)
    parser.add_argument("--target-col",   default=None)
    parser.add_argument("--results-dir",  default="./results")
    parser.add_argument("--plots-dir",    default="./plots")
    parser.add_argument("--n-bootstrap",  type=int, default=None)
    parser.add_argument("--experiments",  nargs="+",
                        choices=["ensemble_size", "depth", "noise", "train_size"],
                        default=["ensemble_size", "depth", "noise", "train_size"])
    parser.add_argument("--quick",        action="store_true",
                        help="Reduce sweep values for fast testing")
    parser.add_argument("--no-plots",     action="store_true")
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("--val-size",     type=int, default=1000,
                        help="Validation set size (0 disables tuning).")
    parser.add_argument("--no-tune",      action="store_true",
                        help="Skip val-set hyperparameter tuning.")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Step 0: Global setup
# ─────────────────────────────────────────────

def setup(args: argparse.Namespace) -> None:
    np.random.seed(GLOBAL_SEED)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir,   exist_ok=True)
    _banner("Bagging vs Boosting · Bias-Variance Pipeline")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Results dir: {args.results_dir}")
    print(f"  Plots dir  : {args.plots_dir}")
    print(f"  Experiments: {args.experiments}")
    n_boot = args.n_bootstrap or BOOTSTRAP_CONFIG["n_bootstrap"]
    print(f"  Bootstrap  : {n_boot} runs")
    print(f"  Quick mode : {args.quick}")
    print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


# ─────────────────────────────────────────────
# Step 1: Build data layer
# ─────────────────────────────────────────────

def build_data_layer(args: argparse.Namespace):
    _step("1", "Building data layer")

    # ── Data generator ────────────────────────────────────────────────────
    if args.dataset == "synthetic":
        generator = SyntheticDataGenerator(
            n_samples=DATASET_CONFIG["n_total"],
            n_features=DATASET_CONFIG["n_features"],
            n_informative=DATASET_CONFIG["n_informative"],
            noise_level=DATASET_CONFIG["default_noise"],
            random_seed=GLOBAL_SEED,
        )
    elif args.dataset == "california":
        generator = CaliforniaHousingDataGenerator()
    elif args.dataset == "csv":
        if not args.csv_path or not args.target_col:
            print("ERROR: --csv-path and --target-col required for csv dataset.")
            sys.exit(1)
        generator = CustomCSVDataGenerator(args.csv_path, args.target_col)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    info = generator.get_info()
    print(f"  Generator  : {info['type']}")
    print(f"  n_samples  : {info['n_samples']}")

    # ── Preprocessor ──────────────────────────────────────────────────────
    cache_dir = os.path.join(args.results_dir, "cache") if args.dataset != "csv" else None
    val_size = 0 if args.no_tune else max(0, args.val_size)
    preprocessor = DataPreprocessor(
        data_generator=generator,
        test_size=DATASET_CONFIG["n_test"],
        val_size=val_size,
        random_seed=GLOBAL_SEED,
        cache_dir=cache_dir,
    )
    preprocessor.initialise()

    X_train, X_test, y_train, y_test = preprocessor.get_train_test_split()
    if preprocessor.has_val_split:
        print(f"  Train pool : {X_train.shape}  Val: {preprocessor.X_val.shape}  Test: {X_test.shape}")
    else:
        print(f"  Train pool : {X_train.shape}  Test: {X_test.shape}  (no val split)")

    # ── Variant manager ───────────────────────────────────────────────────
    variant_manager = DatasetVariantManager(preprocessor, generator)

    noise_levels = EXPERIMENT_VARIATIONS["noise"]["values"]
    train_sizes  = EXPERIMENT_VARIATIONS["train_size"]["values"]

    # For non-synthetic datasets, noise variants are skipped (no-op)
    variant_manager.create_noise_variants(noise_levels)
    variant_manager.create_size_variants(train_sizes)

    print(f"  Variants   : {variant_manager.list_variants()}")
    _ok()
    return preprocessor, variant_manager, generator, info


# ─────────────────────────────────────────────
# Step 2: Model configuration
# ─────────────────────────────────────────────

def build_model_layer(args: argparse.Namespace) -> HyperparameterRegistry:
    _step("2", "Building model configuration layer")
    registry = HyperparameterRegistry()

    if args.quick:
        # Override sweep values for fast testing
        _patch_registry_for_quick_mode(registry)

    print(f"  Models     : RF, XGBoost, Bagged-XGB")
    print(f"  Variations : {list(EXPERIMENT_VARIATIONS.keys())}")
    _ok()
    return registry


# ─────────────────────────────────────────────
# Step 2.5: Hyperparameter tuning
# ─────────────────────────────────────────────

def run_hyperparameter_tuning(
    args: argparse.Namespace,
    preprocessor: DataPreprocessor,
    registry: HyperparameterRegistry,
    results_store: ResultsStore,
) -> Optional[Dict[str, Dict]]:
    """
    Grid-search HPs for RF, XGB, SVR on the val set ONCE.
    Tuned configs replace registry baselines, so all subsequent bootstrap
    runs use the same fixed tuned configuration.

    Persists tuned configs to <results_dir>/tuned_params.json for
    reproducibility and resumability.
    """
    if args.no_tune or not preprocessor.has_val_split:
        if args.no_tune:
            print("  --no-tune set, skipping hyperparameter tuning.")
        else:
            print("  No validation split available, skipping tuning.")
        return None

    _step("2.5", "Hyperparameter tuning (val-set grid search)")

    # Resumability: load cached tuned params if present
    tuned_path = os.path.join(args.results_dir, "tuned_params.json")
    if os.path.exists(tuned_path):
        import json
        with open(tuned_path) as f:
            tuned = json.load(f)
        print(f"  Loaded cached tuned params from {tuned_path}")
        _apply_tuned_to_registry(tuned, registry)
        _print_tuned_summary(tuned)
        _ok()
        return tuned

    tuner = HyperparameterTuner(
        X_train=preprocessor._X_train_raw,
        y_train=preprocessor._y_train,
        X_val=preprocessor.X_val,
        y_val=preprocessor.y_val,
        verbose=True,
    )

    baseline_params = {
        "rf":         registry.get_baseline_config("rf"),
        "xgb":        registry.get_baseline_config("xgb"),
        "svr":        registry.get_baseline_config("svr"),
        "bagged_xgb": registry.get_baseline_config("bagged_xgb"),
    }

    tuned = tuner.tune_all(
        baseline_params=baseline_params,
        max_n_estimators=500,
        early_stopping_rounds=20,
    )

    # Apply to registry so downstream code sees tuned baselines
    _apply_tuned_to_registry(tuned, registry)

    # Persist (JSON-safe: convert sets/None preserved via json)
    import json
    with open(tuned_path, "w") as f:
        json.dump(tuned, f, indent=2, default=str)
    print(f"  Tuned params saved → {tuned_path}")

    _ok()
    return tuned


def _apply_tuned_to_registry(tuned: Dict[str, Dict], registry: HyperparameterRegistry):
    """Mutate the registry's internal baselines with tuned configs."""
    for model_type, cfg in tuned.items():
        registry._baseline[model_type] = copy.deepcopy(cfg)


def _print_tuned_summary(tuned: Dict[str, Dict]):
    for mt in ("rf", "xgb", "svr"):
        if mt in tuned:
            print(f"    {mt}: {tuned[mt]}")


def _patch_registry_for_quick_mode(registry: HyperparameterRegistry):
    """Shrink sweep values for fast smoke-testing."""
    from config import EXPERIMENT_VARIATIONS
    # Rebuild variations with reduced values
    registry._variations.clear()
    for val in [5, 50]:
        registry._add_raw("ensemble_size", "rf",  "n_estimators", val)
        registry._add_raw("ensemble_size", "xgb", "n_estimators", val)
        registry._add_raw("ensemble_size", "bagged_xgb", "outer_n_estimators", val)
    for val in [2, 10]:
        registry._add_raw("depth", "rf",  "max_depth", val)
        registry._add_raw("depth", "xgb", "max_depth", val)
        registry._add_raw("depth", "bagged_xgb", "inner_max_depth", val)
    for sigma in [5, 50]:
        registry._add_raw("noise", "rf",  "_dataset_noise", sigma)
        registry._add_raw("noise", "xgb", "_dataset_noise", sigma)
        registry._add_raw("noise", "bagged_xgb", "_dataset_noise", sigma)
    for n in [500, 4000]:
        registry._add_raw("train_size", "rf",  "_dataset_size", n)
        registry._add_raw("train_size", "xgb", "_dataset_size", n)
        registry._add_raw("train_size", "bagged_xgb", "_dataset_size", n)
    print("  ⚡ Quick mode: reduced sweep values")


# ─────────────────────────────────────────────
# Step 3: Experiments E1–E4
# ─────────────────────────────────────────────

def run_experiments(
    args: argparse.Namespace,
    registry: HyperparameterRegistry,
    variant_manager: DatasetVariantManager,
    results_store: ResultsStore,
) -> Dict[str, ExperimentResult]:
    _step("3", "Running experiments E1–E4")

    n_boot = args.n_bootstrap or BOOTSTRAP_CONFIG["n_bootstrap"]
    evaluator = BootstrapEvaluator(
        n_bootstrap=n_boot,
        random_seed=GLOBAL_SEED,
        verbose=args.verbose,
    )
    orchestrator = ExperimentOrchestrator(
        registry=registry,
        evaluator=evaluator,
        results_store=results_store,
        verbose=True,
    )

    experiments: Dict[str, ExperimentResult] = {}
    t0 = time.time()

    for variation_name in args.experiments:
        experiments[variation_name] = orchestrator.run_experiment(
            variation_name=variation_name,
            variant_manager=variant_manager,
        )

    elapsed = time.time() - t0
    total_configs = sum(len(e.config_results) for e in experiments.values())
    print(f"\n  ✓ All experiments complete: {total_configs} configs  ({elapsed:.1f}s total)")
    _ok()
    return experiments


# ─────────────────────────────────────────────
# Step 4a: SVR baseline
# ─────────────────────────────────────────────

# SVR train size cap. Matches E1 train size so that bootstrap samples drawn
# with seed = base_seed + i are identical across RF/XGB/SVR runs, giving
# valid Cov(A, B) estimates from separate evaluate() calls.
SVR_BASELINE_N_TRAIN = 4000


def run_svr_baseline(
    args: argparse.Namespace,
    preprocessor: DataPreprocessor,
    results_store: ResultsStore,
    n_bootstrap: int,
    registry: Optional[HyperparameterRegistry] = None,
) -> Optional[np.ndarray]:
    """
    Train SVR (RBF kernel) baseline and store its (M × n_test) prediction
    matrix in the results store. Returns the prediction matrix.

    Train size is fixed at SVR_BASELINE_N_TRAIN to match E1 — required for
    valid covariance estimates against RF and XGB (same bootstrap seed →
    same bootstrap rows iff the same training pool size is used).
    """
    _step("4a", "SVR baseline evaluation")

    # Quick smoke-test: shrink train size further to keep runtime tractable
    n_svr = SVR_BASELINE_N_TRAIN
    if args.quick:
        n_svr = min(n_svr, 1000)

    print(f"  SVR train size: {n_svr}  (matches E1 for valid covariance)")

    # Resumability: check if SVR baseline predictions are already on disk
    cached = results_store.get_predictions("svr", "svr_baseline", 0)
    if cached is not None and len(cached):
        print(f"  Using cached SVR baseline predictions: shape={cached.shape}")
        _ok()
        return cached

    # Use tuned SVR config if registry has been updated, else fall back to BASELINE_PARAMS
    svr_cfg = registry.get_baseline_config("svr") if registry is not None else BASELINE_PARAMS["svr"]
    print(f"  SVR config: {svr_cfg}")
    svr_model = ModelBuilder.build("svr", svr_cfg)

    X_tr = preprocessor.X_test  # placeholder, overwritten below
    X_tr, y_tr = preprocessor.get_train_subset(n_svr)

    evaluator = BootstrapEvaluator(
        n_bootstrap=n_bootstrap,
        random_seed=GLOBAL_SEED,
        verbose=args.verbose,
    )
    bv = evaluator.evaluate(
        svr_model,
        X_tr, y_tr,
        preprocessor.X_test, preprocessor.y_test,
        label="SVR baseline",
    )

    cr = ConfigResult(
        model_type="svr",
        variation_name="svr_baseline",
        parameter_name="baseline",
        parameter_value=0,
        bias_squared=bv.bias_squared,
        variance=bv.variance,
        mse=bv.mse,
        all_predictions=bv.all_predictions,
        mean_prediction=bv.mean_prediction,
    )
    results_store.save_config_result(cr)

    print(
        f"  SVR: bias²={bv.bias_squared:.4f}  var={bv.variance:.4f}  mse={bv.mse:.4f}"
    )
    _ok()
    return bv.all_predictions


# ─────────────────────────────────────────────
# Step 4: Hybrid analysis E5–E6 — three pairs
# ─────────────────────────────────────────────

# Hybrid pairs: (pair_name, model_a, model_b)
HYBRID_PAIRS = [
    ("rf_xgb",  "rf",  "xgb"),
    ("rf_svr",  "rf",  "svr"),
    ("xgb_svr", "xgb", "svr"),
]


def run_hybrid_analysis(
    args: argparse.Namespace,
    experiments: Dict[str, ExperimentResult],
    preprocessor: DataPreprocessor,
    results_store: ResultsStore,
    registry: HyperparameterRegistry,
    svr_preds: Optional[np.ndarray] = None,
) -> Dict[str, Dict]:
    """
    For each hybrid pair (rf_xgb, rf_svr, xgb_svr), retrieve A and B baseline
    predictions, compute convex and probabilistic hybrids, and store results.

    Returns
    -------
    Dict keyed by pair_name:
        {pair_name: {"convex": ..., "prob": ..., "stats": ..., "engine": ...,
                     "preds_a": ..., "preds_b": ...}}
    """
    _step("4", "Running hybrid analysis (E5–E6) — 3 pairs")

    y_test = preprocessor.y_test

    baseline_val = _get_baseline_value("ensemble_size", registry)
    print(f"  Using baseline ensemble_size = {baseline_val}")

    # Look up per-model baseline prediction matrices once
    preds_by_model: Dict[str, Optional[np.ndarray]] = {
        "rf":  results_store.get_predictions("rf",  "ensemble_size", baseline_val),
        "xgb": results_store.get_predictions("xgb", "ensemble_size", baseline_val),
        "svr": svr_preds if svr_preds is not None
               else results_store.get_predictions("svr", "svr_baseline", 0),
    }

    engine = HybridComputationEngine(random_seed=GLOBAL_SEED)
    n_runs = BOOTSTRAP_CONFIG["n_prob_hybrid_runs"]

    pair_results: Dict[str, Dict] = {}

    for pair_name, model_a, model_b in HYBRID_PAIRS:
        preds_a = preds_by_model.get(model_a)
        preds_b = preds_by_model.get(model_b)

        if preds_a is None or preds_b is None:
            print(f"  ⚠ Skipping pair {pair_name}: predictions missing "
                  f"(a={model_a}, b={model_b})")
            continue

        # Some configurations may produce mismatched M (e.g. quick-mode SVR
        # with the same n_bootstrap). Trim to common M just in case.
        M_common = min(preds_a.shape[0], preds_b.shape[0])
        preds_a = preds_a[:M_common]
        preds_b = preds_b[:M_common]

        print(f"  [{pair_name}] computing convex + probabilistic hybrids...")
        convex = engine.compute_convex_hybrid(
            preds_a, preds_b, y_test, LAMBDA_VALUES, pair_name=pair_name,
        )
        prob = engine.compute_probabilistic_hybrid(
            preds_a, preds_b, y_test, LAMBDA_VALUES,
            n_runs=n_runs, pair_name=pair_name,
        )
        stats = engine.compute_all_statistics(preds_a, preds_b, y_test)

        results_store.save_hybrid_result(f"convex_{pair_name}", convex)
        results_store.save_hybrid_result(f"prob_{pair_name}",   prob)

        print(
            f"    A({model_a}): bias²={stats['bias_a']:.4f} var={stats['var_a']:.4f} mse={stats['mse_a']:.4f}\n"
            f"    B({model_b}): bias²={stats['bias_b']:.4f} var={stats['var_b']:.4f} mse={stats['mse_b']:.4f}\n"
            f"    Cov(A,B)={stats['covariance']:.4f}  ρ={stats['rho']:.4f}  D²={stats['bias_diff_sq']:.4f}"
        )

        pair_results[pair_name] = {
            "convex":  convex,
            "prob":    prob,
            "stats":   stats,
            "engine":  engine,
            "preds_a": preds_a,
            "preds_b": preds_b,
            "model_a": model_a,
            "model_b": model_b,
        }

    _ok()
    return pair_results


# ─────────────────────────────────────────────
# Step 5: Theory validation
# ─────────────────────────────────────────────

def run_theory_validation(
    args: argparse.Namespace,
    hybrid_pairs: Dict[str, Dict],
    experiments: Dict[str, ExperimentResult],
    preprocessor: DataPreprocessor,
    results_store: ResultsStore,
    registry: HyperparameterRegistry,
) -> List[TheoryValidationRow]:
    """
    Multi-pair theory validation:
      - rf_xgb: full conditions (baseline, low_noise, high_noise, small_data, large_data).
      - rf_svr / xgb_svr: 'baseline' only (SVR has no noise/size variants).
    """
    _step("5", "Theory validation (λ* derivation) — multi-pair")

    if not hybrid_pairs:
        print("  ⚠ No hybrid data available. Skipping theory validation.")
        return []

    y_test = preprocessor.y_test
    engine = next(iter(hybrid_pairs.values()))["engine"]

    theory_rows: List[TheoryValidationRow] = []

    baseline_val = _get_baseline_value("ensemble_size", registry)

    # ── RF + XGB: full set of conditions ─────────────────────────────────
    rf_xgb_conditions = {
        "baseline":   ("ensemble_size", baseline_val),
        "low_noise":  ("noise",  5),
        "high_noise": ("noise",  50),
        "small_data": ("train_size", 500),
        "large_data": ("train_size", 8000),
    }

    if "rf_xgb" in hybrid_pairs:
        for condition_name, (variation, param_val) in rf_xgb_conditions.items():
            if variation not in experiments:
                print(f"  ⚠ Skipping rf_xgb/{condition_name}: experiment '{variation}' not run.")
                continue

            rf_preds  = results_store.get_predictions("rf",  variation, param_val)
            xgb_preds = results_store.get_predictions("xgb", variation, param_val)

            if rf_preds is None or xgb_preds is None:
                print(f"  ⚠ Skipping rf_xgb/{condition_name}: predictions not found.")
                continue

            stats = engine.compute_all_statistics(rf_preds, xgb_preds, y_test)
            convex = engine.compute_convex_hybrid(
                rf_preds, xgb_preds, y_test, LAMBDA_VALUES, pair_name="rf_xgb",
            )
            prob = engine.compute_probabilistic_hybrid(
                rf_preds, xgb_preds, y_test, LAMBDA_VALUES,
                n_runs=BOOTSTRAP_CONFIG["n_prob_hybrid_runs"],
                pair_name="rf_xgb",
            )

            row = TheoryValidator.build_validation_row(
                condition_name=condition_name,
                pair_name="rf_xgb",
                stats=stats,
                convex_results=convex,
                prob_results=prob,
            )
            theory_rows.append(row)
            results_store.save_theory_row(row)

            print(
                f"  rf_xgb · {condition_name:11s} | "
                f"λ*_conv: theory={row.lambda_star_convex_theory:.3f}  "
                f"empirical={row.lambda_star_convex_empirical:.3f}  "
                f"|diff|={abs(row.lambda_star_convex_theory - row.lambda_star_convex_empirical):.3f}"
            )

    # ── Cross-family pairs: baseline only ────────────────────────────────
    for pair_name in ("rf_svr", "xgb_svr"):
        if pair_name not in hybrid_pairs:
            continue

        payload = hybrid_pairs[pair_name]
        stats  = payload["stats"]
        convex = payload["convex"]
        prob   = payload["prob"]

        row = TheoryValidator.build_validation_row(
            condition_name="baseline",
            pair_name=pair_name,
            stats=stats,
            convex_results=convex,
            prob_results=prob,
        )
        theory_rows.append(row)
        results_store.save_theory_row(row)

        print(
            f"  {pair_name} · baseline    | "
            f"λ*_conv: theory={row.lambda_star_convex_theory:.3f}  "
            f"empirical={row.lambda_star_convex_empirical:.3f}  "
            f"|diff|={abs(row.lambda_star_convex_theory - row.lambda_star_convex_empirical):.3f}"
        )

    # Print summary table
    df = results_store.theory_rows_to_dataframe()
    if not df.empty:
        print("\n  Theory Validation Table:")
        print(df.to_string(index=False))

    _ok()
    return theory_rows


# ─────────────────────────────────────────────
# Step 6: Analysis tables
# ─────────────────────────────────────────────

def run_analysis(
    args: argparse.Namespace,
    experiments: Dict[str, ExperimentResult],
    hybrid_pairs: Dict[str, Dict],
    theory_rows: List[TheoryValidationRow],
    results_store: ResultsStore,
) -> None:
    _step("6", "Building analysis tables")

    table_builder = TableBuilder(results_store)

    # ── Theory validation table ───────────────────────────────────────────
    theory_df = table_builder.build_theory_validation_table()
    if not theory_df.empty:
        theory_path = os.path.join(args.results_dir, "theory_validation.csv")
        theory_df.to_csv(theory_path, index=False)
        print(f"  Theory table → {theory_path}")

    # ── Hypothesis verdict table ──────────────────────────────────────────
    verdict_df = table_builder.build_hypothesis_verdict_table(
        experiments=experiments,
        hybrid_pairs=hybrid_pairs,
        theory_rows=theory_rows,
    )
    verdict_path = os.path.join(args.results_dir, "hypothesis_verdicts.csv")
    verdict_df.to_csv(verdict_path, index=False)
    print(f"  Verdict table → {verdict_path}")

    # ── Full results DataFrame ────────────────────────────────────────────
    all_results_df = results_store.all_config_results_to_dataframe()
    all_path = os.path.join(args.results_dir, "all_results.csv")
    all_results_df.to_csv(all_path, index=False)
    print(f"  All results  → {all_path}")

    # Print hypothesis table to stdout
    print("\n  Hypothesis Verdicts:")
    cols = ["Hypothesis", "Observed", "Verdict"]
    print(verdict_df[cols].to_string(index=False))

    _ok()


# ─────────────────────────────────────────────
# Step 7: Plots
# ─────────────────────────────────────────────

def run_visualization(
    args: argparse.Namespace,
    experiments: Dict[str, ExperimentResult],
    hybrid_pairs: Dict[str, Dict],
    theory_rows: List[TheoryValidationRow],
    results_store: ResultsStore,
    preprocessor: DataPreprocessor,
) -> None:
    if args.no_plots:
        print("  (--no-plots set, skipping visualization)")
        return

    _step("7", "Generating plots")

    trajectory_plotter = TrajectoryPlotter(results_store, dpi=150)
    hybrid_plotter     = HybridPlotter(dpi=150)

    # ── Plots 1-4: Trajectory atlas ───────────────────────────────────────
    plot_methods = {
        "ensemble_size": trajectory_plotter.plot_ensemble_size,
        "depth":         trajectory_plotter.plot_depth,
        "noise":         trajectory_plotter.plot_noise,
        "train_size":    trajectory_plotter.plot_train_size,
    }
    for variation_name, method in plot_methods.items():
        if variation_name not in experiments:
            continue
        save_path = os.path.join(args.plots_dir, f"plot_{variation_name}.png")
        fig = method(save_path=save_path)
        plt_close(fig)
        print(f"  Plot saved → {save_path}")

    # ── Plots 5-6: Hybrid analysis (RF + XGB pair, primary) ──────────────
    rf_xgb_payload = hybrid_pairs.get("rf_xgb")
    if rf_xgb_payload:
        convex = rf_xgb_payload.get("convex", {})
        prob   = rf_xgb_payload.get("prob",   {})
        stats  = rf_xgb_payload.get("stats",  {})

        baseline_row = next(
            (r for r in theory_rows
             if r.condition_name == "baseline" and r.pair_name == "rf_xgb"),
            None,
        )

        if convex and prob and baseline_row:
            from pipeline_types import BiasVarianceResult
            rf_bv = BiasVarianceResult(
                bias_squared=stats["bias_a"],
                variance=stats["var_a"],
                mse=stats["mse_a"],
                mean_prediction=np.array([]),
                all_predictions=np.array([]),
            )
            xgb_bv = BiasVarianceResult(
                bias_squared=stats["bias_b"],
                variance=stats["var_b"],
                mse=stats["mse_b"],
                mean_prediction=np.array([]),
                all_predictions=np.array([]),
            )
            save5 = os.path.join(args.plots_dir, "plot_hybrid_bv_space.png")
            fig5 = hybrid_plotter.plot_hybrid_bv_space(
                rf_bv=rf_bv,
                xgb_bv=xgb_bv,
                convex_results=convex,
                prob_results=prob,
                lambda_star_convex=baseline_row.lambda_star_convex_empirical,
                lambda_star_prob=baseline_row.lambda_star_prob_empirical,
                save_path=save5,
            )
            plt_close(fig5)
            print(f"  Plot saved → {save5}")

            save6 = os.path.join(args.plots_dir, "plot_mse_vs_lambda.png")
            fig6 = hybrid_plotter.plot_mse_vs_lambda(
                convex_results=convex,
                prob_results=prob,
                lambda_star_convex_theory=baseline_row.lambda_star_convex_theory,
                lambda_star_prob_theory=baseline_row.lambda_star_prob_theory,
                lambda_star_convex_empirical=baseline_row.lambda_star_convex_empirical,
                lambda_star_prob_empirical=baseline_row.lambda_star_prob_empirical,
                rf_mse=stats["mse_a"],
                xgb_mse=stats["mse_b"],
                save_path=save6,
            )
            plt_close(fig6)
            print(f"  Plot saved → {save6}")

    # ── Plot 7: ρ bar chart across pairs ─────────────────────────────────
    if hybrid_pairs:
        pair_rhos = {
            p: payload["stats"]["rho"]
            for p, payload in hybrid_pairs.items()
            if "stats" in payload and "rho" in payload["stats"]
        }
        if pair_rhos:
            pair_labels = {
                p: f"{MODEL_LABELS.get(payload['model_a'], payload['model_a'])} + "
                   f"{MODEL_LABELS.get(payload['model_b'], payload['model_b'])}"
                for p, payload in hybrid_pairs.items() if p in pair_rhos
            }
            save7 = os.path.join(args.plots_dir, "plot_correlation_barchart.png")
            fig7 = hybrid_plotter.plot_correlation_barchart(
                pair_rhos=pair_rhos,
                pair_labels=pair_labels,
                save_path=save7,
            )
            plt_close(fig7)
            print(f"  Plot saved → {save7}")

        # ── Plot 8: Multi-pair MSE vs λ (convex) ──────────────────────────
        pairs_convex = {
            p: payload["convex"]
            for p, payload in hybrid_pairs.items()
            if payload.get("convex")
        }
        if pairs_convex:
            pair_labels_8 = {
                p: f"{MODEL_LABELS.get(payload['model_a'], payload['model_a'])} + "
                   f"{MODEL_LABELS.get(payload['model_b'], payload['model_b'])}"
                for p, payload in hybrid_pairs.items() if p in pairs_convex
            }
            save8 = os.path.join(args.plots_dir, "plot_mse_vs_lambda_multipair.png")
            fig8 = hybrid_plotter.plot_mse_vs_lambda_multipair(
                pairs_convex=pairs_convex,
                pair_labels=pair_labels_8,
                save_path=save8,
            )
            plt_close(fig8)
            print(f"  Plot saved → {save8}")

    _ok()


def plt_close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def _get_baseline_value(variation_name: str, registry: HyperparameterRegistry):
    """Return the last (largest/most complete) parameter value for a model in a variation."""
    vals = registry.list_variation_values(variation_name, "rf")
    if not vals:
        return None
    # Filter None, return largest
    numeric = [v for v in vals if v is not None]
    return max(numeric) if numeric else vals[-1]


def _banner(text: str):
    width = 64
    print("═" * width)
    pad = (width - len(text) - 2) // 2
    print(f"{'═' * pad}  {text}  {'═' * (width - pad - len(text) - 2)}")
    print("═" * width)


def _step(n: str, text: str):
    print(f"\n{'─'*64}")
    print(f"  Step {n}: {text}")
    print(f"{'─'*64}")


def _ok():
    print("  ─ Done ✓")


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    setup(args)

    # ── Build infrastructure ──────────────────────────────────────────────
    results_store = ResultsStore(output_dir=args.results_dir)

    # ── Step 1: Data ──────────────────────────────────────────────────────
    preprocessor, variant_manager, generator, dataset_info = build_data_layer(args)

    # ── Save metadata ─────────────────────────────────────────────────────
    results_store.save_metadata({
        "dataset":    dataset_info,
        "timestamp":  datetime.now().isoformat(),
        "n_bootstrap": args.n_bootstrap or BOOTSTRAP_CONFIG["n_bootstrap"],
        "experiments": args.experiments,
        "quick_mode":  args.quick,
    })

    # ── Step 2: Models ────────────────────────────────────────────────────
    registry = build_model_layer(args)

    # ── Step 2.5: Hyperparameter tuning (val-set grid search) ─────────────
    run_hyperparameter_tuning(args, preprocessor, registry, results_store)

    # ── Step 3: Experiments E1-E4 ─────────────────────────────────────────
    experiments = run_experiments(args, registry, variant_manager, results_store)

    # ── Step 4a: SVR baseline (mixing partner) ────────────────────────────
    n_boot = args.n_bootstrap or BOOTSTRAP_CONFIG["n_bootstrap"]
    svr_preds = None
    if "ensemble_size" in experiments:
        svr_preds = run_svr_baseline(args, preprocessor, results_store, n_boot, registry=registry)

    # ── Step 4: Hybrid E5-E6 (3 pairs) ───────────────────────────────────
    hybrid_pairs: Dict[str, Dict] = {}
    if "ensemble_size" in experiments:
        hybrid_pairs = run_hybrid_analysis(
            args, experiments, preprocessor, results_store, registry,
            svr_preds=svr_preds,
        )

    # ── Step 5: Theory validation ─────────────────────────────────────────
    theory_rows = run_theory_validation(
        args, hybrid_pairs, experiments, preprocessor, results_store, registry
    )

    # ── Step 6: Analysis tables ───────────────────────────────────────────
    run_analysis(args, experiments, hybrid_pairs, theory_rows, results_store)

    # ── Step 7: Plots ─────────────────────────────────────────────────────
    run_visualization(
        args, experiments, hybrid_pairs, theory_rows, results_store, preprocessor
    )

    # ── Done ──────────────────────────────────────────────────────────────
    _banner("Pipeline complete")
    print(f"  Results → {args.results_dir}")
    print(f"  Plots   → {args.plots_dir}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == "__main__":
    main()
