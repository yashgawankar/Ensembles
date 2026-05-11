"""
types.py — Shared dataclasses and typed containers used across all pipeline layers.
Keeping types in one place avoids circular imports and keeps the codebase explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ─────────────────────────────────────────────
# Core bias-variance result for one model configuration
# ─────────────────────────────────────────────

@dataclass
class BiasVarianceResult:
    """Output of a single bootstrap evaluation run."""
    bias_squared: float
    variance: float
    mse: float
    mean_prediction: np.ndarray            # shape: (n_test,)
    all_predictions: np.ndarray            # shape: (n_bootstrap, n_test)

    # Optional: store per-point estimates for richer analysis
    per_point_bias_sq: Optional[np.ndarray] = None   # shape: (n_test,)
    per_point_variance: Optional[np.ndarray] = None  # shape: (n_test,)


# ─────────────────────────────────────────────
# Result for one specific parameter configuration
# ─────────────────────────────────────────────

@dataclass
class ConfigResult:
    """
    Records the output of one (model, variation, parameter_value) triplet.
    E.g., RF with ensemble_size = 50.
    """
    model_type: str           # 'rf', 'xgb', 'bagged_xgb'
    variation_name: str       # 'ensemble_size', 'depth', 'noise', 'train_size'
    parameter_name: str       # e.g. 'n_estimators'
    parameter_value: object   # e.g. 50 or None

    bias_squared: float
    variance: float
    mse: float

    all_predictions: np.ndarray   # shape: (n_bootstrap, n_test)
    mean_prediction: np.ndarray   # shape: (n_test,)

    # Key used in ResultsStore
    @property
    def key(self) -> str:
        return f"{self.model_type}__{self.variation_name}__{self.parameter_value}"


# ─────────────────────────────────────────────
# Full experiment result (one variation sweep)
# ─────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """
    Aggregates all ConfigResults for one experiment variation.
    E.g., E1 = all (RF, XGB, Bagged-XGB) × ensemble_size values.
    """
    variation_name: str
    label: str
    config_results: List[ConfigResult] = field(default_factory=list)

    def get(self, model_type: str, parameter_value: object) -> Optional[ConfigResult]:
        for cr in self.config_results:
            if cr.model_type == model_type and cr.parameter_value == parameter_value:
                return cr
        return None

    def get_model_trajectory(self, model_type: str) -> Tuple[List, List, List]:
        """Returns (param_values, bias_sq_list, variance_list) for one model, sorted by param."""
        results = sorted(
            [cr for cr in self.config_results if cr.model_type == model_type],
            key=lambda cr: (cr.parameter_value is None, cr.parameter_value)
        )
        param_vals = [cr.parameter_value for cr in results]
        bias_sqs   = [cr.bias_squared    for cr in results]
        variances  = [cr.variance         for cr in results]
        return param_vals, bias_sqs, variances


# ─────────────────────────────────────────────
# Hybrid result for one λ value
# ─────────────────────────────────────────────

@dataclass
class HybridResult:
    """Result of the convex or probabilistic hybrid at a specific λ."""
    lambda_val: float
    hybrid_type: str       # 'convex' or 'probabilistic'

    bias_squared: float
    variance: float
    mse: float

    all_predictions: np.ndarray   # shape: (n_bootstrap, n_test)

    pair_name: str = ""    # e.g. "rf_xgb", "rf_svr", "xgb_svr"


# ─────────────────────────────────────────────
# Theory validation row
# ─────────────────────────────────────────────

@dataclass
class TheoryValidationRow:
    condition_name: str
    pair_name: str          # e.g. "rf_xgb", "rf_svr", "xgb_svr"

    bias_a: float
    var_a: float
    bias_b: float
    var_b: float
    covariance: float
    rho: float              # Pearson correlation of predictions

    lambda_star_convex_theory: float
    lambda_star_convex_empirical: float

    lambda_star_prob_theory: float
    lambda_star_prob_empirical: float

    mse_a: float
    mse_b: float
    mse_convex_at_star: float
    mse_prob_at_star: float


# ─────────────────────────────────────────────
# Hypothesis verdict
# ─────────────────────────────────────────────

@dataclass
class HypothesisVerdict:
    hypothesis_id: str        # 'H1', 'H2', ...
    description: str
    prediction: str
    observed: str
    verdict: str              # 'Supported', 'Partially Supported', 'Not Supported'
