"""
experiments/orchestrator.py — ExperimentOrchestrator.

Runs experiments E1–E4 by iterating over all (model, variation, value)
configurations, delegating to BootstrapEvaluator, and persisting results
to ResultsStore.

Key responsibilities:
  - Select the correct dataset variant (full vs noise_* vs size_*)
  - Build each model via ModelBuilder
  - Run BootstrapEvaluator
  - Wrap results in ConfigResult / ExperimentResult
  - Save to ResultsStore
  - Provide resumability (skip already-computed configs)
"""

from __future__ import annotations

import time
from typing import List, Optional

from tqdm import tqdm

from data.preprocessor import DatasetVariantManager
from experiments.bootstrap_evaluator import BootstrapEvaluator
from models.model_builder import HyperparameterRegistry, ModelBuilder
from analysis.results_store import ResultsStore
from pipeline_types import ConfigResult, ExperimentResult


class ExperimentOrchestrator:
    """
    High-level coordinator for experiments E1–E4.

    Usage
    -----
    orchestrator = ExperimentOrchestrator(registry, evaluator, results_store)
    result = orchestrator.run_experiment('ensemble_size', variant_manager)
    """

    def __init__(
        self,
        registry: HyperparameterRegistry,
        evaluator: BootstrapEvaluator,
        results_store: ResultsStore,
        verbose: bool = True,
    ):
        self.registry = registry
        self.evaluator = evaluator
        self.results_store = results_store
        self.verbose = verbose

    # ── Main API ──────────────────────────────────────────────────────────

    def run_experiment(
        self,
        variation_name: str,
        variant_manager: DatasetVariantManager,
        model_types: Optional[List[str]] = None,
    ) -> ExperimentResult:
        """
        Run one complete experiment sweep (e.g., E1: ensemble size).

        Parameters
        ----------
        variation_name : One of 'ensemble_size', 'depth', 'noise', 'train_size'.
        variant_manager: DatasetVariantManager with noise + size variants pre-created.
        model_types    : Optional filter (default: all three models).

        Returns
        -------
        ExperimentResult containing all ConfigResults.
        """
        from config import EXPERIMENT_VARIATIONS

        exp_info = EXPERIMENT_VARIATIONS[variation_name]
        label = exp_info["label"]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Experiment: {label}  [{variation_name}]")
            print(f"{'='*60}")

        # Get all (model_type, param_name, param_value, config) tuples
        all_configs = self.registry.list_all_configs(variation_name)

        # Filter by model_types if specified
        if model_types:
            all_configs = [c for c in all_configs if c[0] in model_types]

        experiment = ExperimentResult(
            variation_name=variation_name,
            label=label,
            config_results=[],
        )

        outer_bar = tqdm(
            all_configs,
            desc=f"[{label}]",
            unit="config",
            dynamic_ncols=True,
        )

        for model_type, param_name, param_value, config in outer_bar:
            run_label = f"{model_type} | {param_name}={param_value}"
            outer_bar.set_postfix_str(run_label)

            # ── Resumability: skip if already computed ─────────────────────
            existing = self.results_store.get_config_result(
                model_type, variation_name, param_value
            )
            if existing is not None:
                tqdm.write(f"  ↩  SKIP (cached): {run_label}")
                experiment.config_results.append(existing)
                continue

            # ── Select correct dataset variant ─────────────────────────────
            variant_key = self._resolve_variant_key(variation_name, param_name, param_value)
            X_train, X_test, y_train, y_test = variant_manager.get_variant(variant_key)

            # ── Build model and evaluate ───────────────────────────────────
            t0 = time.time()
            model = ModelBuilder.build(model_type, config)
            bv_result = self.evaluator.evaluate(
                model, X_train, y_train, X_test, y_test,
                label=run_label,
            )
            elapsed = time.time() - t0

            tqdm.write(
                f"  ✓  {run_label:<45} "
                f"bias²={bv_result.bias_squared:>10.4f}  "
                f"var={bv_result.variance:>10.4f}  "
                f"mse={bv_result.mse:>10.4f}  "
                f"({elapsed:.1f}s)"
            )

            # ── Wrap and store ─────────────────────────────────────────────
            config_result = ConfigResult(
                model_type=model_type,
                variation_name=variation_name,
                parameter_name=param_name,
                parameter_value=param_value,
                bias_squared=bv_result.bias_squared,
                variance=bv_result.variance,
                mse=bv_result.mse,
                all_predictions=bv_result.all_predictions,
                mean_prediction=bv_result.mean_prediction,
            )

            self.results_store.save_config_result(config_result)
            experiment.config_results.append(config_result)

        # Persist full experiment summary
        self.results_store.save_experiment_result(experiment)

        tqdm.write(f"\n  ✓ {label} complete — {len(experiment.config_results)} configs\n")

        return experiment

    def run_all_experiments(
        self,
        variant_manager: DatasetVariantManager,
    ) -> dict[str, ExperimentResult]:
        """
        Convenience method: runs E1–E4 in order.

        Returns
        -------
        Dict mapping variation_name → ExperimentResult.
        """
        experiments = {}
        for variation_name in ["ensemble_size", "depth", "noise", "train_size"]:
            experiments[variation_name] = self.run_experiment(
                variation_name, variant_manager
            )
        return experiments

    # ── Internal helpers ──────────────────────────────────────────────────

    def _resolve_variant_key(
        self,
        variation_name: str,
        param_name: str,
        param_value,
    ) -> str:
        """
        Map (variation_name, param_name, param_value) → dataset variant key.

        For HP variations (ensemble size, depth): use the full training pool.
        For dataset variations (noise, size): select the appropriate variant.
        """
        if param_name == "_dataset_noise":
            return f"noise_{param_value}"
        elif param_name == "_dataset_size":
            return f"size_{param_value}"
        else:
            # HP variation → use the full baseline training split
            return "full"
