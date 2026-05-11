"""
experiments/hybrid_engine.py — HybridComputationEngine.

Computes convex and probabilistic hybrids from stored prediction matrices for
arbitrary pairs of models (A and B). No new model training occurs here – we
reuse the (M × n_test) prediction matrices already stored in ResultsStore.

Mathematical definitions
────────────────────────

Convex hybrid:
    ŷ_λ[i, j] = λ · preds_b[i, j] + (1-λ) · preds_a[i, j]

Probabilistic hybrid:
    For each (bootstrap run i, test point j):
        r ~ Uniform(0, 1)
        ŷ_λ[i, j] = preds_b[i, j] if r < λ else preds_a[i, j]

Bias² / variance computed identically for both:
    mean_pred[j]  = mean over i of ŷ_λ[:, j]
    bias²         = mean over j of (mean_pred[j] - y_test[j])²
    variance      = mean over j of Var_i(ŷ_λ[:, j])
    MSE           = bias² + variance

Covariance (for λ* derivation):
    cov[j] = Cov_i(preds_a[:, j], preds_b[:, j])
    covariance = mean over j of cov[j]
"""

from __future__ import annotations
import _path_setup

from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline_types import HybridResult


class HybridComputationEngine:
    """
    Computes bias²/variance/MSE for convex and probabilistic hybrids
    over a sweep of λ values, using pre-computed prediction matrices.

    Models are referred to generically as A (the λ=0 endpoint) and
    B (the λ=1 endpoint).
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    # ── Public API ────────────────────────────────────────────────────────

    def compute_convex_hybrid(
        self,
        preds_a: np.ndarray,             # (M, n_test)
        preds_b: np.ndarray,             # (M, n_test)
        y_test: np.ndarray,              # (n_test,)
        lambda_values: List[float],
        pair_name: str = "",
    ) -> Dict[float, HybridResult]:
        """
        For each λ, compute:
            hybrid = λ · preds_b + (1-λ) · preds_a
        and return bias², variance, MSE.
        """
        results: Dict[float, HybridResult] = {}

        for lam in lambda_values:
            lam = round(lam, 10)
            hybrid_preds = lam * preds_b + (1.0 - lam) * preds_a
            bv = self._compute_bv(hybrid_preds, y_test)

            results[lam] = HybridResult(
                lambda_val=lam,
                hybrid_type="convex",
                bias_squared=bv["bias_squared"],
                variance=bv["variance"],
                mse=bv["mse"],
                all_predictions=hybrid_preds,
                pair_name=pair_name,
            )

        return results

    def compute_probabilistic_hybrid(
        self,
        preds_a: np.ndarray,             # (M, n_test)
        preds_b: np.ndarray,             # (M, n_test)
        y_test: np.ndarray,
        lambda_values: List[float],
        n_runs: int = 100,
        pair_name: str = "",
    ) -> Dict[float, HybridResult]:
        """
        For each λ, simulate probabilistic model selection:
            P(use B | λ) = λ
            P(use A | λ) = 1 - λ

        The randomness is per-prediction-per-bootstrap-run.  To get stable
        variance estimates we average the resulting prediction matrix over
        n_runs independent randomisations.

        The averaged prediction matrix is stored in HybridResult for downstream
        analysis, but note it represents E[ŷ_λ] and not individual samples.
        For bias²/variance we use the average across runs.
        """
        results: Dict[float, HybridResult] = {}

        M, n_test = preds_a.shape

        for lam in lambda_values:
            lam = round(lam, 10)

            # Edge cases: λ=0 ≡ A, λ=1 ≡ B
            if lam == 0.0:
                hybrid_preds = preds_a.copy()
            elif lam == 1.0:
                hybrid_preds = preds_b.copy()
            else:
                # Average n_runs draws to reduce Monte-Carlo noise
                accumulated = np.zeros((M, n_test))
                for run in range(n_runs):
                    rng = np.random.RandomState(self.random_seed + run + int(lam * 1000))
                    mask = rng.uniform(0, 1, size=(M, n_test)) < lam  # True → use B
                    draw = np.where(mask, preds_b, preds_a)
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
                pair_name=pair_name,
            )

        return results

    def compute_covariance(
        self,
        preds_a: np.ndarray,    # (M, n_test)
        preds_b: np.ndarray,    # (M, n_test)
    ) -> float:
        """
        Compute the average per-point covariance between A and B
        predictions across bootstrap runs.

        cov[j] = Cov_i(preds_a[:, j], preds_b[:, j])
        covariance = mean_j(cov[j])

        This is a scalar estimator used in the theoretical λ* derivations.
        """
        a_centered = preds_a - preds_a.mean(axis=0, keepdims=True)
        b_centered = preds_b - preds_b.mean(axis=0, keepdims=True)

        per_point_cov = (a_centered * b_centered).mean(axis=0)  # (n_test,)
        return float(per_point_cov.mean())

    def compute_all_statistics(
        self,
        preds_a: np.ndarray,
        preds_b: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """
        Convenience method: compute all statistics needed for theory validation.

        Returns
        -------
        dict with keys:
            bias_a, var_a, mse_a            (squared bias, variance, MSE for model A)
            bias_b, var_b, mse_b            (same for model B)
            signed_bias_a, signed_bias_b    (mean residual — sign preserved)
            covariance                      (Cov(A, B) averaged over test points)
            bias_diff_sq                    (E[(μ_A - μ_B)²] — switching penalty)
            rho                             (Pearson correlation across bootstrap runs)
        """
        bv_a = self._compute_bv(preds_a, y_test)
        bv_b = self._compute_bv(preds_b, y_test)
        cov  = self.compute_covariance(preds_a, preds_b)

        # TRUE signed biases (fix: was np.sqrt which is always positive,
        # losing sign information needed for the B0·Δb cross-term in λ*)
        signed_bias_a = float((preds_a.mean(axis=0) - y_test).mean())
        signed_bias_b = float((preds_b.mean(axis=0) - y_test).mean())

        # Pearson correlation of predictions across bootstrap runs
        rho = cov / (np.sqrt(bv_a["variance"]) * np.sqrt(bv_b["variance"]) + 1e-12)

        # Expected prediction difference (scalar, used in probabilistic λ*)
        mean_a = preds_a.mean(axis=0)
        mean_b = preds_b.mean(axis=0)
        bias_diff_sq = float(((mean_b - mean_a) ** 2).mean())

        return {
            "bias_a":        bv_a["bias_squared"],
            "var_a":         bv_a["variance"],
            "mse_a":         bv_a["mse"],
            "bias_b":        bv_b["bias_squared"],
            "var_b":         bv_b["variance"],
            "mse_b":         bv_b["mse"],
            "signed_bias_a": signed_bias_a,
            "signed_bias_b": signed_bias_b,
            "covariance":    cov,
            "bias_diff_sq":  bias_diff_sq,
            "rho":           float(rho),
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _compute_bv(predictions: np.ndarray, y_test: np.ndarray) -> Dict:
        """Decompose predictions matrix into bias², variance, MSE."""
        mean_pred  = predictions.mean(axis=0)
        bias_sq    = float(((mean_pred - y_test) ** 2).mean())
        variance   = float(predictions.var(axis=0, ddof=0).mean())
        return {"bias_squared": bias_sq, "variance": variance, "mse": bias_sq + variance}
