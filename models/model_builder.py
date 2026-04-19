"""
models/model_builder.py — HyperparameterRegistry and ModelBuilder.

HyperparameterRegistry is the central store for all model configs.
ModelBuilder is a pure factory – it returns *unfitted* estimators.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("xgboost is required: pip install xgboost")

from config import BASELINE_PARAMS, EXPERIMENT_VARIATIONS, GLOBAL_SEED


# ─────────────────────────────────────────────
# HyperparameterRegistry
# ─────────────────────────────────────────────

class HyperparameterRegistry:
    """
    Central store for all model hyperparameter configurations.

    Usage:
        registry = HyperparameterRegistry()
        configs = registry.list_all_configs('ensemble_size')
        # → list of (model_type, param_name, param_value, config_dict)
    """

    def __init__(self):
        self._baseline: Dict[str, Dict] = copy.deepcopy(BASELINE_PARAMS)
        self._variations: Dict[str, List[Tuple[str, str, Any]]] = {}
        # (variation_name → list of (model_type, param_name, param_value))

        # Auto-populate from config.py
        self._build_default_variations()

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_default_variations(self):
        """
        Translate EXPERIMENT_VARIATIONS from config.py into internal registry entries.
        """

        # ── E1: Ensemble size ─────────────────────────────────────────────
        for val in EXPERIMENT_VARIATIONS["ensemble_size"]["values"]:
            self._add_raw("ensemble_size", "rf",  "n_estimators", val)
            self._add_raw("ensemble_size", "xgb", "n_estimators", val)
            self._add_raw("ensemble_size", "bagged_xgb", "outer_n_estimators", val)

        # ── E2: Depth ──────────────────────────────────────────────────────
        depth_vals = EXPERIMENT_VARIATIONS["depth"]["values"]
        for val in depth_vals:
            self._add_raw("depth", "rf",  "max_depth", val)
            self._add_raw("depth", "xgb", "max_depth", val)
            self._add_raw("depth", "bagged_xgb", "inner_max_depth", val)

        # RF also gets unlimited depth (None)
        for val in EXPERIMENT_VARIATIONS["depth"].get("values_rf_extra", []):
            self._add_raw("depth", "rf", "max_depth", val)

        # ── E3: Noise – dataset variation, not model HP; mark as passthrough ─
        for sigma in EXPERIMENT_VARIATIONS["noise"]["values"]:
            self._add_raw("noise", "rf",  "_dataset_noise", sigma)
            self._add_raw("noise", "xgb", "_dataset_noise", sigma)
            self._add_raw("noise", "bagged_xgb", "_dataset_noise", sigma)

        # ── E4: Train size – dataset variation ───────────────────────────
        for n in EXPERIMENT_VARIATIONS["train_size"]["values"]:
            self._add_raw("train_size", "rf",  "_dataset_size", n)
            self._add_raw("train_size", "xgb", "_dataset_size", n)
            self._add_raw("train_size", "bagged_xgb", "_dataset_size", n)

    def _add_raw(self, variation: str, model_type: str, param_name: str, param_value: Any):
        if variation not in self._variations:
            self._variations[variation] = []
        self._variations[variation].append((model_type, param_name, param_value))

    # ── Public API ────────────────────────────────────────────────────────

    def get_baseline_config(self, model_type: str) -> Dict:
        """Return a deep copy of the baseline config for a model."""
        return copy.deepcopy(self._baseline[model_type])

    def get_config(
        self,
        model_type: str,
        variation_name: str,
        param_value: Any,
    ) -> Dict:
        """
        Return config for a specific (model, variation, value) triplet.

        If the param is a dataset selector (_dataset_noise / _dataset_size),
        the returned config is just the baseline (dataset selection happens
        in the orchestrator, not in the config dict).
        """
        config = copy.deepcopy(self._baseline[model_type])

        # Find matching entry in the variation list
        entries = [
            (mt, pn, pv)
            for (mt, pn, pv) in self._variations.get(variation_name, [])
            if mt == model_type and pv == param_value
        ]

        if not entries:
            raise KeyError(
                f"No config found for ({model_type}, {variation_name}, {param_value})"
            )

        _, param_name, _ = entries[0]

        # Dataset variations → no HP change, return baseline as-is
        if param_name.startswith("_dataset"):
            return config

        # Apply the varied parameter
        if model_type in ("rf", "xgb"):
            config[param_name] = param_value

        elif model_type == "bagged_xgb":
            if param_name == "outer_n_estimators":
                config["outer"]["n_estimators"] = param_value
            elif param_name == "inner_max_depth":
                config["inner"]["max_depth"] = param_value
            else:
                config["outer"][param_name] = param_value

        return config

    def list_all_configs(
        self, variation_name: str
    ) -> List[Tuple[str, str, Any, Dict]]:
        """
        Return list of (model_type, param_name, param_value, config_dict)
        for every entry in a given variation.

        Ordered: RF, XGB, Bagged-XGB for each parameter value.
        """
        raw = self._variations.get(variation_name, [])
        results = []
        for (model_type, param_name, param_value) in raw:
            config = self.get_config(model_type, variation_name, param_value)
            results.append((model_type, param_name, param_value, config))
        return results

    def list_variation_values(self, variation_name: str, model_type: str) -> List[Any]:
        """Return all parameter values for a (variation, model) pair."""
        return [
            pv
            for (mt, pn, pv) in self._variations.get(variation_name, [])
            if mt == model_type
        ]


# ─────────────────────────────────────────────
# ModelBuilder
# ─────────────────────────────────────────────

class ModelBuilder:
    """
    Pure factory class.  Takes (model_type, config_dict) → returns unfitted
    sklearn-compatible estimator.  Training happens in BootstrapEvaluator.
    """

    @staticmethod
    def build(model_type: str, config: Dict) -> BaseEstimator:
        """
        Parameters
        ----------
        model_type : 'rf' | 'xgb' | 'bagged_xgb'
        config     : dict returned by HyperparameterRegistry.get_config()

        Returns
        -------
        Unfitted estimator (sklearn BaseEstimator interface).
        """
        if model_type == "rf":
            return ModelBuilder._build_rf(config)
        elif model_type == "xgb":
            return ModelBuilder._build_xgb(config)
        elif model_type == "bagged_xgb":
            return ModelBuilder._build_bagged_xgb(config)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose: rf, xgb, bagged_xgb")

    @staticmethod
    def _build_rf(config: Dict) -> RandomForestRegressor:
        return RandomForestRegressor(**config)

    @staticmethod
    def _build_xgb(config: Dict) -> XGBRegressor:
        # XGBoost does not accept 'random_state' kwarg in older versions; map to seed
        cfg = copy.deepcopy(config)
        if "random_state" in cfg:
            cfg.setdefault("seed", cfg.pop("random_state"))
        return XGBRegressor(**cfg)

    @staticmethod
    def _build_bagged_xgb(config: Dict) -> BaggingRegressor:
        outer_cfg = copy.deepcopy(config["outer"])
        inner_cfg = copy.deepcopy(config["inner"])

        # XGBoost seed mapping
        if "random_state" in inner_cfg:
            inner_cfg.setdefault("seed", inner_cfg.pop("random_state"))

        inner_estimator = XGBRegressor(**inner_cfg)

        # Remove n_jobs from outer if unsupported version of sklearn
        return BaggingRegressor(
            estimator=inner_estimator,
            **outer_cfg,
        )

    @staticmethod
    def build_from_baseline(model_type: str) -> BaseEstimator:
        """Build model using baseline config directly from BASELINE_PARAMS."""
        config = copy.deepcopy(BASELINE_PARAMS[model_type])
        return ModelBuilder.build(model_type, config)
