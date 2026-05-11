"""
analysis/results_store.py — ResultsStore.

Central, structured storage for all pipeline outputs.

Storage strategy:
  - In-memory dict for fast access during a run
  - Disk: JSON for scalar metrics, numpy .npy for prediction matrices
  - Separate subdirectories per artifact type
  - Resumability: check disk before computing

Disk layout
───────────
results/
├── experiments/
│   ├── ensemble_size.json   ← scalar metrics only
│   ├── depth.json
│   ├── noise.json
│   └── train_size.json
├── hybrids/
│   ├── convex_baseline.json
│   └── probabilistic_baseline.json
├── predictions/
│   ├── rf__ensemble_size__50.npy
│   ├── xgb__ensemble_size__50.npy
│   └── ...
└── metadata.json
"""

from __future__ import annotations
import _path_setup

import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline_types import (
    ConfigResult,
    ExperimentResult,
    HybridResult,
    TheoryValidationRow,
)


def _npy_safe_value(v):
    """Convert numpy scalars to Python native for JSON serialisation."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


class ResultsStore:
    """
    Thread-safe (within a single process) store for all pipeline results.
    """

    def __init__(self, output_dir: str = "./results"):
        self.output_dir = output_dir
        self._ensure_dirs()

        # In-memory caches
        self._config_results: Dict[str, ConfigResult]     = {}
        self._experiment_results: Dict[str, ExperimentResult] = {}
        self._hybrid_results: Dict[str, Dict[float, HybridResult]] = {}
        self._theory_rows: List[TheoryValidationRow]      = []

    # ── Directory setup ───────────────────────────────────────────────────

    def _ensure_dirs(self):
        for sub in ("experiments", "hybrids", "predictions", "metadata"):
            os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)

    def _experiments_path(self, variation_name: str) -> str:
        return os.path.join(self.output_dir, "experiments", f"{variation_name}.json")

    def _hybrid_path(self, name: str) -> str:
        return os.path.join(self.output_dir, "hybrids", f"{name}.json")

    def _prediction_path(self, key: str) -> str:
        safe_key = key.replace("/", "_").replace(".", "_")
        return os.path.join(self.output_dir, "predictions", f"{safe_key}.npy")

    def _metadata_path(self) -> str:
        return os.path.join(self.output_dir, "metadata", "metadata.json")

    # ── ConfigResult save / load ──────────────────────────────────────────

    def save_config_result(self, result: ConfigResult):
        """Persist one ConfigResult (scalars to JSON, predictions to .npy)."""
        # Memory
        self._config_results[result.key] = result

        # Predictions matrix
        pred_path = self._prediction_path(result.key)
        np.save(pred_path, result.all_predictions)

    def get_config_result(
        self,
        model_type: str,
        variation_name: str,
        parameter_value,
    ) -> Optional[ConfigResult]:
        """
        Retrieve a ConfigResult, checking memory first then disk.
        Returns None if not found (enabling resumability checks).
        """
        # Build key
        key = f"{model_type}__{variation_name}__{parameter_value}"

        # Memory hit
        if key in self._config_results:
            return self._config_results[key]

        # Disk: check if the experiment JSON has it
        exp_path = self._experiments_path(variation_name)
        if not os.path.exists(exp_path):
            return None

        with open(exp_path) as f:
            data = json.load(f)

        for row in data.get("config_results", []):
            if (
                row["model_type"] == model_type
                and row["variation_name"] == variation_name
                and str(row["parameter_value"]) == str(parameter_value)
            ):
                # Rebuild ConfigResult from disk
                pred_path = self._prediction_path(key)
                all_preds = np.load(pred_path) if os.path.exists(pred_path) else None
                mean_pred = all_preds.mean(axis=0) if all_preds is not None else None

                cr = ConfigResult(
                    model_type=model_type,
                    variation_name=variation_name,
                    parameter_name=row["parameter_name"],
                    parameter_value=row["parameter_value"],
                    bias_squared=row["bias_squared"],
                    variance=row["variance"],
                    mse=row["mse"],
                    all_predictions=all_preds if all_preds is not None else np.array([]),
                    mean_prediction=mean_pred if mean_pred is not None else np.array([]),
                )
                self._config_results[key] = cr
                return cr

        return None

    def get_predictions(
        self,
        model_type: str,
        variation_name: str,
        parameter_value,
    ) -> Optional[np.ndarray]:
        """Load prediction matrix (M × n_test) for a specific config."""
        key = f"{model_type}__{variation_name}__{parameter_value}"
        cr = self.get_config_result(model_type, variation_name, parameter_value)
        if cr is not None and cr.all_predictions is not None and len(cr.all_predictions):
            return cr.all_predictions
        # Try .npy directly
        pred_path = self._prediction_path(key)
        if os.path.exists(pred_path):
            return np.load(pred_path)
        return None

    # ── ExperimentResult save / load ──────────────────────────────────────

    def save_experiment_result(self, result: ExperimentResult):
        """Persist scalar metrics for an entire experiment to JSON."""
        self._experiment_results[result.variation_name] = result

        rows = []
        for cr in result.config_results:
            rows.append({
                "model_type":       cr.model_type,
                "variation_name":   cr.variation_name,
                "parameter_name":   cr.parameter_name,
                "parameter_value":  _npy_safe_value(cr.parameter_value),
                "bias_squared":     _npy_safe_value(cr.bias_squared),
                "variance":         _npy_safe_value(cr.variance),
                "mse":              _npy_safe_value(cr.mse),
            })

        payload = {
            "variation_name": result.variation_name,
            "label":          result.label,
            "config_results": rows,
        }
        with open(self._experiments_path(result.variation_name), "w") as f:
            json.dump(payload, f, indent=2)

    def load_experiment_result(self, variation_name: str) -> Optional[ExperimentResult]:
        """Load an experiment from disk (scalars only; predictions loaded lazily)."""
        path = self._experiments_path(variation_name)
        if not os.path.exists(path):
            return None

        with open(path) as f:
            data = json.load(f)

        config_results = []
        for row in data["config_results"]:
            pred_path = self._prediction_path(
                f"{row['model_type']}__{variation_name}__{row['parameter_value']}"
            )
            all_preds = np.load(pred_path) if os.path.exists(pred_path) else np.array([])
            mean_pred = all_preds.mean(axis=0) if len(all_preds) else np.array([])

            cr = ConfigResult(
                model_type=row["model_type"],
                variation_name=row["variation_name"],
                parameter_name=row["parameter_name"],
                parameter_value=row["parameter_value"],
                bias_squared=row["bias_squared"],
                variance=row["variance"],
                mse=row["mse"],
                all_predictions=all_preds,
                mean_prediction=mean_pred,
            )
            config_results.append(cr)
            self._config_results[cr.key] = cr

        er = ExperimentResult(
            variation_name=variation_name,
            label=data.get("label", variation_name),
            config_results=config_results,
        )
        self._experiment_results[variation_name] = er
        return er

    def get_experiment_result(self, variation_name: str) -> Optional[ExperimentResult]:
        if variation_name in self._experiment_results:
            return self._experiment_results[variation_name]
        return self.load_experiment_result(variation_name)

    # ── Hybrid results save / load ────────────────────────────────────────

    def save_hybrid_result(self, name: str, results: Dict[float, HybridResult]):
        """Persist hybrid results (convex or probabilistic) for one pair/condition."""
        self._hybrid_results[name] = results

        rows = []
        for lam, hr in results.items():
            rows.append({
                "lambda": _npy_safe_value(hr.lambda_val),
                "hybrid_type": hr.hybrid_type,
                "bias_squared": _npy_safe_value(hr.bias_squared),
                "variance": _npy_safe_value(hr.variance),
                "mse": _npy_safe_value(hr.mse),
                "pair_name": hr.pair_name,
            })

        with open(self._hybrid_path(name), "w") as f:
            json.dump({"name": name, "results": rows}, f, indent=2)

    def get_hybrid_result(self, name: str) -> Optional[Dict[float, HybridResult]]:
        if name in self._hybrid_results:
            return self._hybrid_results[name]

        path = self._hybrid_path(name)
        if not os.path.exists(path):
            return None

        with open(path) as f:
            data = json.load(f)

        results = {}
        for row in data["results"]:
            lam = row["lambda"]
            results[lam] = HybridResult(
                lambda_val=lam,
                hybrid_type=row["hybrid_type"],
                bias_squared=row["bias_squared"],
                variance=row["variance"],
                mse=row["mse"],
                all_predictions=np.array([]),  # Not stored on disk
                pair_name=row.get("pair_name", ""),
            )
        self._hybrid_results[name] = results
        return results

    # ── Theory rows ───────────────────────────────────────────────────────

    def save_theory_row(self, row: TheoryValidationRow):
        self._theory_rows.append(row)

    def get_theory_rows(self) -> List[TheoryValidationRow]:
        return list(self._theory_rows)

    # ── DataFrame export ──────────────────────────────────────────────────

    def all_config_results_to_dataframe(self) -> pd.DataFrame:
        """Export all ConfigResult scalars to a flat DataFrame for analysis."""
        rows = []
        for cr in self._config_results.values():
            rows.append({
                "model_type":      cr.model_type,
                "variation_name":  cr.variation_name,
                "parameter_name":  cr.parameter_name,
                "parameter_value": cr.parameter_value,
                "bias_squared":    cr.bias_squared,
                "variance":        cr.variance,
                "mse":             cr.mse,
            })
        return pd.DataFrame(rows)

    def hybrid_results_to_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        """Export hybrid results for one condition to DataFrame."""
        results = self.get_hybrid_result(name)
        if results is None:
            return None
        rows = [
            {
                "lambda":       hr.lambda_val,
                "hybrid_type":  hr.hybrid_type,
                "bias_squared": hr.bias_squared,
                "variance":     hr.variance,
                "mse":          hr.mse,
            }
            for hr in results.values()
        ]
        return pd.DataFrame(rows).sort_values("lambda")

    def theory_rows_to_dataframe(self) -> pd.DataFrame:
        """Export theory validation table to DataFrame."""
        rows = []
        for r in self._theory_rows:
            rows.append({
                "Pair":                      r.pair_name,
                "Condition":                 r.condition_name,
                "Bias² A":                   round(r.bias_a, 4),
                "Var A":                     round(r.var_a, 4),
                "Bias² B":                   round(r.bias_b, 4),
                "Var B":                     round(r.var_b, 4),
                "ρ(A,B)":                    round(r.rho, 4),
                "Cov(A,B)":                  round(r.covariance, 4),
                "λ* Convex (theory)":        round(r.lambda_star_convex_theory, 4),
                "λ* Convex (empirical)":     round(r.lambda_star_convex_empirical, 4),
                "λ* Prob (theory)":          round(r.lambda_star_prob_theory, 4),
                "λ* Prob (empirical)":       round(r.lambda_star_prob_empirical, 4),
                "MSE A":                     round(r.mse_a, 4),
                "MSE B":                     round(r.mse_b, 4),
                "MSE Convex @ λ*":           round(r.mse_convex_at_star, 4),
                "MSE Prob @ λ*":             round(r.mse_prob_at_star, 4),
            })
        return pd.DataFrame(rows)

    # ── Save metadata ─────────────────────────────────────────────────────

    def save_metadata(self, metadata: dict):
        """Save pipeline run metadata (dataset info, config, timestamps)."""
        with open(self._metadata_path(), "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def load_metadata(self) -> Optional[dict]:
        path = self._metadata_path()
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)
