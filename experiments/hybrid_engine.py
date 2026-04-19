"""
experiments/hybrid_engine.py — HybridComputationEngine.

Computes convex and probabilistic hybrids from stored RF + XGB predictions.
No new model training occurs here – we reuse the (M × n_test) prediction
matrices already stored in ResultsStore.

Mathematical definitions
────────────────────────

Convex hybrid:
    ŷ_λ[i, j] = λ · xgb[i, j] + (1-λ) · rf[i, j]

Probabilistic hybrid:
    For each (bootstrap run i, test point j):
        r ~ Uniform(0, 1)
        ŷ_λ[i, j] = xgb[i, j] if r < λ else rf[i, j]

Bias² / variance computed identically for both:
    mean_pred[j]  = mean over i of ŷ_λ[:, j]
    bias²         = mean over j of (mean_pred[j] - y_test[j])²
    variance      = mean over j of Var_i(ŷ_λ[:, j])
    MSE           = bias² + variance

Covariance (for λ* derivation):
    cov[j] = Cov_i(rf[:, j], xgb[:, j])
    covariance = mean over j of cov[j]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline_types import HybridResult


class HybridComputationEngine:
    """
    Computes bias²/variance/MSE for convex and probabilistic hybrids
    over a sweep of λ values, using pre-computed prediction matrices.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    # ── Public API ────────────────────────────────────────────────────────

    def compute_convex_hybrid(
        self,
        rf_predictions: np.ndarray,      # (M, n_test)
        xgb_predictions: np.ndarray,     # (M, n_test)
        y_test: np.ndarray,              # (n_test,)
        lambda_values: List[float],
    ) -> Dict[float, HybridResult]:
        """
        For each λ, compute:
            hybrid = λ · xgb + (1-λ) · rf
        and return bias², variance, MSE.
        """
        results: Dict[float, HybridResult] = {}

        for lam in lambda_values:
            lam = round(lam, 10)
            hybrid_preds = lam * xgb_predictions + (1.0 - lam) * rf_predictions
            bv = self._compute_bv(hybrid_preds, y_test)

            results[lam] = HybridResult(
                lambda_val=lam,
                hybrid_type="convex",
                bias_squared=bv["bias_squared"],
                variance=bv["variance"],
                mse=bv["mse"],
                all_predictions=hybrid_preds,
            )

        return results

    def compute_probabilistic_hybrid(
        self,
        rf_predictions: np.ndarray,      # (M, n_test)
        xgb_predictions: np.ndarray,     # (M, n_test)
        y_test: np.ndarray,
        lambda_values: List[float],
        n_runs: int = 100,
    ) -> Dict[float, HybridResult]:
        """
        For each λ, simulate probabilistic model selection:
            P(use XGB | λ) = λ
            P(use RF  | λ) = 1 - λ

        The randomness is per-prediction-per-bootstrap-run.  To get stable
        variance estimates we average the resulting prediction matrix over
        n_runs independent randomisations.

        The averaged prediction matrix is stored in HybridResult for downstream
        analysis, but note it represents E[ŷ_λ] and not individual samples.
        For bias²/variance we use the average across runs.
        """
        results: Dict[float, HybridResult] = {}

        M, n_test = rf_predictions.shape

        for lam in lambda_values:
            lam = round(lam, 10)

            # Edge cases: λ=0 ≡ RF, λ=1 ≡ XGB
            if lam == 0.0:
                hybrid_preds = rf_predictions.copy()
            elif lam == 1.0:
                hybrid_preds = xgb_predictions.copy()
            else:
                # Average n_runs draws to reduce Monte-Carlo noise
                accumulated = np.zeros((M, n_test))
                for run in range(n_runs):
                    rng = np.random.RandomState(self.random_seed + run + int(lam * 1000))
                    mask = rng.uniform(0, 1, size=(M, n_test)) < lam  # True → use XGB
                    draw = np.where(mask, xgb_predictions, rf_predictions)
                    accumulated += draw
                hybrid_preds = accumulated / n_runs

            bv = self._compute_bv(hybrid_preds, y_test)

            results[lam] = HybridResult(
                lambda_val=lam,
                hybrid_type="probabilistic",
                bias_squared=bv["bias_squared"],
                variance=bv["variance"],
                mse=bv["mse"],
                all_predictions=hybrid_preds,
            )

        return results

    def compute_covariance(
        self,
        rf_predictions: np.ndarray,    # (M, n_test)
        xgb_predictions: np.ndarray,   # (M, n_test)
    ) -> float:
        """
        Compute the average per-point covariance between RF and XGB
        predictions across bootstrap runs.

        cov[j] = Cov_i(rf[:, j], xgb[:, j])
        covariance = mean_j(cov[j])

        This is a scalar estimator used in the theoretical λ* derivations.
        """
        M = rf_predictions.shape[0]

        rf_centered  = rf_predictions  - rf_predictions.mean(axis=0, keepdims=True)
        xgb_centered = xgb_predictions - xgb_predictions.mean(axis=0, keepdims=True)

        per_point_cov = (rf_centered * xgb_centered).mean(axis=0)  # (n_test,)
        return float(per_point_cov.mean())

    def compute_all_statistics(
        self,
        rf_predictions: np.ndarray,
        xgb_predictions: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """
        Convenience method: compute all statistics needed for theory validation.

        Returns
        -------
        dict with keys:
            bias_rf, var_rf, mse_rf
            bias_xgb, var_xgb, mse_xgb
            covariance
            bias_diff_sq  (for probabilistic λ* formula)
        """
        bv_rf  = self._compute_bv(rf_predictions, y_test)
        bv_xgb = self._compute_bv(xgb_predictions, y_test)
        cov    = self.compute_covariance(rf_predictions, xgb_predictions)

        # Expected prediction difference (scalar, used in probabilistic λ*)
        mean_rf  = rf_predictions.mean(axis=0)
        mean_xgb = xgb_predictions.mean(axis=0)
        bias_diff_sq = float(((mean_xgb - mean_rf) ** 2).mean())

        return {
            "bias_rf":      bv_rf["bias_squared"],
            "var_rf":       bv_rf["variance"],
            "mse_rf":       bv_rf["mse"],
            "bias_xgb":     bv_xgb["bias_squared"],
            "var_xgb":      bv_xgb["variance"],
            "mse_xgb":      bv_xgb["mse"],
            "covariance":   cov,
            "bias_diff_sq": bias_diff_sq,
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _compute_bv(predictions: np.ndarray, y_test: np.ndarray) -> Dict:
        """Decompose predictions matrix into bias², variance, MSE."""
        mean_pred  = predictions.mean(axis=0)
        bias_sq    = float(((mean_pred - y_test) ** 2).mean())
        variance   = float(predictions.var(axis=0, ddof=0).mean())
        return {"bias_squared": bias_sq, "variance": variance, "mse": bias_sq + variance}
