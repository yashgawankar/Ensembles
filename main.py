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
    EXPERIMENT_VARIATIONS,
    LAMBDA_VALUES,
    THEORY_CONDITIONS,
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
from analysis.results_store import ResultsStore
from analysis.metrics import MetricsComputer, TableBuilder
from visualization.trajectory_plotter import TrajectoryPlotter
from visualization.hybrid_plotter import HybridPlotter
from pipeline_types import ExperimentResult, TheoryValidationRow


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
    preprocessor = DataPreprocessor(
        data_generator=generator,
        test_size=DATASET_CONFIG["n_test"],
        random_seed=GLOBAL_SEED,
        cache_dir=cache_dir,
    )
    preprocessor.initialise()

    X_train, X_test, y_train, y_test = preprocessor.get_train_test_split()
    print(f"  Train pool : {X_train.shape}  Test: {X_test.shape}")

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
# Step 4: Hybrid analysis E5–E6
# ─────────────────────────────────────────────

def run_hybrid_analysis(
    args: argparse.Namespace,
    experiments: Dict[str, ExperimentResult],
    preprocessor: DataPreprocessor,
    results_store: ResultsStore,
    registry: HyperparameterRegistry,
) -> Dict:
    """
    Retrieve baseline RF + XGB predictions, compute convex and probabilistic
    hybrids for all λ values, and store results.

    Returns a dict with hybrid results and statistics for theory validation.
    """
    _step("4", "Running hybrid analysis (E5–E6)")

    y_test = preprocessor.y_test

    # ── Get baseline predictions ──────────────────────────────────────────
    # We use the ensemble_size experiment at baseline n_estimators=100
    # (or largest available if quick mode)
    baseline_val = _get_baseline_value("ensemble_size", registry)
    print(f"  Using baseline ensemble_size = {baseline_val}")

    rf_preds  = results_store.get_predictions("rf",  "ensemble_size", baseline_val)
    xgb_preds = results_store.get_predictions("xgb", "ensemble_size", baseline_val)

    if rf_preds is None or xgb_preds is None:
        print("  ⚠ Baseline predictions not found in store. Skipping hybrid analysis.")
        return {}

    engine = HybridComputationEngine(random_seed=GLOBAL_SEED)

    # ── Convex hybrid ─────────────────────────────────────────────────────
    print("  Computing convex hybrid...")
    convex = engine.compute_convex_hybrid(rf_preds, xgb_preds, y_test, LAMBDA_VALUES)
    results_store.save_hybrid_result("convex_baseline", convex)

    # ── Probabilistic hybrid ──────────────────────────────────────────────
    print("  Computing probabilistic hybrid...")
    n_runs = BOOTSTRAP_CONFIG["n_prob_hybrid_runs"]
    prob = engine.compute_probabilistic_hybrid(
        rf_preds, xgb_preds, y_test, LAMBDA_VALUES, n_runs=n_runs
    )
    results_store.save_hybrid_result("probabilistic_baseline", prob)

    # ── Core statistics for theory ────────────────────────────────────────
    stats = engine.compute_all_statistics(rf_preds, xgb_preds, y_test)

    print(
        f"  RF:  bias²={stats['bias_rf']:.4f}  var={stats['var_rf']:.4f}  mse={stats['mse_rf']:.4f}"
    )
    print(
        f"  XGB: bias²={stats['bias_xgb']:.4f}  var={stats['var_xgb']:.4f}  mse={stats['mse_xgb']:.4f}"
    )
    print(f"  Cov(RF,XGB) = {stats['covariance']:.4f}")

    _ok()
    return {
        "convex":  convex,
        "prob":    prob,
        "stats":   stats,
        "engine":  engine,
        "rf_preds": rf_preds,
        "xgb_preds": xgb_preds,
    }


# ─────────────────────────────────────────────
# Step 5: Theory validation
# ─────────────────────────────────────────────

def run_theory_validation(
    args: argparse.Namespace,
    hybrid_data: Dict,
    experiments: Dict[str, ExperimentResult],
    preprocessor: DataPreprocessor,
    results_store: ResultsStore,
    registry: HyperparameterRegistry,
) -> List[TheoryValidationRow]:
    _step("5", "Theory validation (λ* derivation)")

    if not hybrid_data:
        print("  ⚠ No hybrid data available. Skipping theory validation.")
        return []

    engine: HybridComputationEngine = hybrid_data["engine"]
    y_test = preprocessor.y_test

    theory_rows: List[TheoryValidationRow] = []

    # Build condition → (rf_preds, xgb_preds) mapping
    # We pull predictions for each theory condition from the results store
    conditions_to_run = {
        "baseline":   ("ensemble_size", _get_baseline_value("ensemble_size", registry)),
        "low_noise":  ("noise",  5),
        "high_noise": ("noise",  50),
        "small_data": ("train_size", 500),
        "large_data": ("train_size", 8000),
    }

    for condition_name, (variation, param_val) in conditions_to_run.items():
        if variation not in experiments:
            print(f"  ⚠ Skipping {condition_name}: experiment '{variation}' not run.")
            continue

        rf_preds  = results_store.get_predictions("rf",  variation, param_val)
        xgb_preds = results_store.get_predictions("xgb", variation, param_val)

        if rf_preds is None or xgb_preds is None:
            print(f"  ⚠ Skipping {condition_name}: predictions not found "
                  f"(variation={variation}, val={param_val})")
            continue

        # Compute stats for this condition
        stats = engine.compute_all_statistics(rf_preds, xgb_preds, y_test)

        # Compute hybrid results for this condition
        convex = engine.compute_convex_hybrid(rf_preds, xgb_preds, y_test, LAMBDA_VALUES)
        prob   = engine.compute_probabilistic_hybrid(
            rf_preds, xgb_preds, y_test, LAMBDA_VALUES,
            n_runs=BOOTSTRAP_CONFIG["n_prob_hybrid_runs"]
        )

        # Build theory validation row
        row = TheoryValidator.build_validation_row(
            condition_name=condition_name,
            stats=stats,
            convex_results=convex,
            prob_results=prob,
        )
        theory_rows.append(row)
        results_store.save_theory_row(row)

        print(
            f"  {condition_name:12s} | "
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
    hybrid_data: Dict,
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
        hybrid_convex=hybrid_data.get("convex"),
        hybrid_prob=hybrid_data.get("prob"),
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
    hybrid_data: Dict,
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

    # ── Plots 5-6: Hybrid analysis ────────────────────────────────────────
    if hybrid_data:
        convex = hybrid_data.get("convex", {})
        prob   = hybrid_data.get("prob",   {})
        stats  = hybrid_data.get("stats",  {})

        # Find empirical λ* from stored results
        baseline_row = next(
            (r for r in theory_rows if r.condition_name == "baseline"), None
        )

        if convex and prob and baseline_row:
            # Plot 5
            from pipeline_types import BiasVarianceResult
            rf_bv = BiasVarianceResult(
                bias_squared=stats["bias_rf"],
                variance=stats["var_rf"],
                mse=stats["mse_rf"],
                mean_prediction=np.array([]),
                all_predictions=np.array([]),
            )
            xgb_bv = BiasVarianceResult(
                bias_squared=stats["bias_xgb"],
                variance=stats["var_xgb"],
                mse=stats["mse_xgb"],
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

            # Plot 6
            save6 = os.path.join(args.plots_dir, "plot_mse_vs_lambda.png")
            fig6 = hybrid_plotter.plot_mse_vs_lambda(
                convex_results=convex,
                prob_results=prob,
                lambda_star_convex_theory=baseline_row.lambda_star_convex_theory,
                lambda_star_prob_theory=baseline_row.lambda_star_prob_theory,
                lambda_star_convex_empirical=baseline_row.lambda_star_convex_empirical,
                lambda_star_prob_empirical=baseline_row.lambda_star_prob_empirical,
                rf_mse=stats["mse_rf"],
                xgb_mse=stats["mse_xgb"],
                save_path=save6,
            )
            plt_close(fig6)
            print(f"  Plot saved → {save6}")

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

    # ── Step 3: Experiments E1-E4 ─────────────────────────────────────────
    experiments = run_experiments(args, registry, variant_manager, results_store)

    # ── Step 4: Hybrid E5-E6 ─────────────────────────────────────────────
    hybrid_data = {}
    if "ensemble_size" in experiments:
        hybrid_data = run_hybrid_analysis(
            args, experiments, preprocessor, results_store, registry
        )

    # ── Step 5: Theory validation ─────────────────────────────────────────
    theory_rows = run_theory_validation(
        args, hybrid_data, experiments, preprocessor, results_store, registry
    )

    # ── Step 6: Analysis tables ───────────────────────────────────────────
    run_analysis(args, experiments, hybrid_data, theory_rows, results_store)

    # ── Step 7: Plots ─────────────────────────────────────────────────────
    run_visualization(
        args, experiments, hybrid_data, theory_rows, results_store, preprocessor
    )

    # ── Done ──────────────────────────────────────────────────────────────
    _banner("Pipeline complete")
    print(f"  Results → {args.results_dir}")
    print(f"  Plots   → {args.plots_dir}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == "__main__":
    main()
