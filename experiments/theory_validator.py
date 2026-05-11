"""
experiments/theory_validator.py — TheoryValidator.

Derives the theoretically optimal mixing weight λ* for both the convex
and probabilistic hybrid from first principles (MSE minimisation).

Mathematical derivations
════════════════════════

── Convex hybrid ────────────────────────────────────────────────────────────

    f_λ = λ · f_XGB + (1-λ) · f_RF

    E[f_λ] = λ · μ_XGB + (1-λ) · μ_RF          where μ = mean prediction

    Bias(f_λ) = E[f_λ] - y = λ · b_XGB + (1-λ) · b_RF
    (b = signed bias; we work with squared bias in MSE)

    Let B  = bias of f_λ  (linear in λ)
        B  = λ·b_XGB + (1-λ)·b_RF
        B² = [λ·b_XGB + (1-λ)·b_RF]²

    Var(f_λ) = λ²·V_XGB + (1-λ)²·V_RF + 2λ(1-λ)·Cov(RF, XGB)

    MSE(λ) = B² + Var(f_λ)

    d MSE / dλ = 0  →
        2[λ·b_XGB + (1-λ)·b_RF]·(b_XGB - b_RF)
        + 2λ·V_XGB - 2(1-λ)·V_RF + 2(1-2λ)·Cov
        = 0

    Letting:
        Δb = b_XGB - b_RF
        B0 = b_RF  (bias at λ=0)

    Expanding and collecting λ terms:

        λ [Δb² + V_XGB + V_RF - 2·Cov]
        + B0·Δb + V_RF·(-1) + Cov·(-1) +  ... wait – full expansion:

    B² = (B0 + λ·Δb)²  = B0² + 2B0·Δb·λ + Δb²·λ²
    d(B²)/dλ = 2B0·Δb + 2Δb²·λ

    Var = λ²·V_XGB + (1-λ)²·V_RF + 2λ(1-λ)·Cov
        = λ²(V_XGB + V_RF - 2Cov) + λ(2Cov - 2V_RF) + V_RF
    d(Var)/dλ = 2λ(V_XGB + V_RF - 2Cov) + (2Cov - 2V_RF)

    Setting sum to zero:
        2B0·Δb + 2Δb²·λ + 2λ(V_XGB + V_RF - 2Cov) + (2Cov - 2V_RF) = 0

        λ [Δb² + V_XGB + V_RF - 2Cov] = V_RF - Cov - B0·Δb

        λ* = (V_RF - Cov - B0·Δb) / (Δb² + V_XGB + V_RF - 2Cov)

    Where:
        B0  = signed bias of RF     = sqrt(bias_sq_rf) · sign(residual_rf)
        Δb  = b_XGB - b_RF
        Note: we approximate B0·Δb ≈ bias_rf - covariance_of_biases, but
        for a scalar estimator from bootstrap means we use the signed version.

── Probabilistic hybrid ─────────────────────────────────────────────────────

    Bias (same as convex):
        B = λ·b_XGB + (1-λ)·b_RF

    Variance (switching adds extra variance):
        Var(f_λ) = λ·V_XGB + (1-λ)·V_RF + λ(1-λ)·(μ_XGB - μ_RF)²

        The last term is the variance induced by randomly choosing between
        models with different expected predictions.

    d MSE / dλ = 0:

        V term d/dλ = V_XGB - V_RF + (1-2λ)·D²   where D² = (μ_XGB - μ_RF)²

        Setting full derivative to zero:
        2B0·Δb + 2Δb²·λ + (V_XGB - V_RF) + (1-2λ)·D² = 0

        λ [2Δb² - 2D²] = V_RF - V_XGB - D² - 2B0·Δb + D²·... let's be careful:

        d/dλ of [λ(1-λ)D²] = (1-2λ)D²

        Full derivative:
        2B0·Δb + 2Δb²·λ + V_XGB - V_RF + D² - 2λD² = 0

        λ [2Δb² - 2D²] = V_RF - V_XGB - D² - 2B0·Δb

        λ* = (V_RF - V_XGB - D² - 2B0·Δb) / (2Δb² - 2D²)

        Special case: if denominator → 0 (Δb = ±D), λ* doesn't exist in
        (0,1) and we clip to [0,1].
"""

from __future__ import annotations
import _path_setup

from typing import Dict, Optional

import numpy as np

from pipeline_types import TheoryValidationRow


class TheoryValidator:
    """
    Analytical and empirical λ* computation + comparison.

    All methods are static – no instance state needed.
    """

    # ── λ* derivations ────────────────────────────────────────────────────

    @staticmethod
    def derive_lambda_star_convex(
        bias_sq_rf: float,
        var_rf: float,
        bias_sq_xgb: float,
        var_xgb: float,
        covariance: float,
        signed_bias_rf: Optional[float] = None,
        signed_bias_xgb: Optional[float] = None,
    ) -> float:
        """
        Analytical λ* for the convex hybrid.

        Parameters (empirically estimated from bootstrap runs)
        ----------
        bias_sq_rf / bias_sq_xgb : scalar bias² for each model
        var_rf / var_xgb          : scalar variance for each model
        covariance                : Cov(RF preds, XGB preds) across bootstrap runs
        signed_bias_*             : If provided, use for Δb·B0 cross-term.
                                    If None, approximate as sqrt(bias_sq) with
                                    positive sign (conservative).

        Returns
        -------
        λ* clipped to [0, 1].
        """
        # Signed biases (if not provided, assume both positive – conservative)
        B0  = signed_bias_rf  if signed_bias_rf  is not None else np.sqrt(bias_sq_rf)
        Bx  = signed_bias_xgb if signed_bias_xgb is not None else np.sqrt(bias_sq_xgb)
        delta_b = Bx - B0

        denom = delta_b**2 + var_xgb + var_rf - 2.0 * covariance

        if abs(denom) < 1e-12:
            # Degenerate case: models are perfectly correlated, same bias → λ arbitrary
            return 0.5

        numer = var_rf - covariance - B0 * delta_b
        lam_star = numer / denom

        return float(np.clip(lam_star, 0.0, 1.0))

    @staticmethod
    def derive_lambda_star_probabilistic(
        bias_sq_rf: float,
        var_rf: float,
        bias_sq_xgb: float,
        var_xgb: float,
        bias_diff_sq: float,
        signed_bias_rf: Optional[float] = None,
        signed_bias_xgb: Optional[float] = None,
    ) -> float:
        """
        Analytical λ* for the probabilistic hybrid.

        bias_diff_sq : E[(μ_XGB - μ_RF)²] averaged over test points –
                       computed by HybridComputationEngine.compute_all_statistics().
        """
        B0      = signed_bias_rf  if signed_bias_rf  is not None else np.sqrt(bias_sq_rf)
        Bx      = signed_bias_xgb if signed_bias_xgb is not None else np.sqrt(bias_sq_xgb)
        delta_b = Bx - B0

        D_sq    = bias_diff_sq  # (μ_XGB - μ_RF)² averaged over test points

        denom = 2.0 * (delta_b**2 - D_sq)

        if abs(denom) < 1e-12:
            # No unique minimum; fall back to MSE-weighted blend
            if var_rf + bias_sq_rf < var_xgb + bias_sq_xgb:
                return 0.0
            return 1.0

        numer = var_rf - var_xgb - D_sq - 2.0 * B0 * delta_b
        lam_star = numer / denom

        return float(np.clip(lam_star, 0.0, 1.0))

    # ── Empirical λ* from curve ───────────────────────────────────────────

    @staticmethod
    def find_empirical_lambda_star(
        hybrid_results: Dict[float, "HybridResult"],
    ) -> float:
        """
        Find λ that achieves minimum MSE among all evaluated λ values.
        Returns the λ value (not the MSE).
        """
        best_lam = min(hybrid_results, key=lambda lam: hybrid_results[lam].mse)
        return float(best_lam)

    # ── Comparison ────────────────────────────────────────────────────────

    @staticmethod
    def compare_theory_vs_empirical(
        lambda_star_theory: float,
        lambda_star_empirical: float,
    ) -> Dict:
        """
        Return comparison statistics between theoretical and empirical λ*.
        """
        abs_diff = abs(lambda_star_theory - lambda_star_empirical)
        is_close = abs_diff <= 0.1
        return {
            "theory":     round(lambda_star_theory, 4),
            "empirical":  round(lambda_star_empirical, 4),
            "abs_diff":   round(abs_diff, 4),
            "is_close":   is_close,
        }

    # ── Theory validation table ───────────────────────────────────────────

    @staticmethod
    def build_validation_row(
        condition_name: str,
        pair_name: str,
        stats: Dict,
        convex_results: Dict[float, "HybridResult"],
        prob_results: Dict[float, "HybridResult"],
    ) -> TheoryValidationRow:
        """
        Build a complete TheoryValidationRow for one experimental condition
        and a specific (A, B) model pair.

        Parameters
        ----------
        condition_name : e.g. 'baseline', 'low_noise', ...
        pair_name      : e.g. 'rf_xgb', 'rf_svr', 'xgb_svr'
        stats          : output of HybridComputationEngine.compute_all_statistics()
                         Must contain: bias_a, var_a, bias_b, var_b, covariance,
                         signed_bias_a, signed_bias_b, bias_diff_sq, rho, mse_a, mse_b.
        convex_results : output of HybridComputationEngine.compute_convex_hybrid()
        prob_results   : output of HybridComputationEngine.compute_probabilistic_hybrid()
        """
        # FIX: use TRUE signed biases (mean residual). Previous code used
        # np.sqrt(bias_sq) which is always positive and lost sign information,
        # causing the B0·Δb cross-term in λ* to skew the estimator.
        b_a = stats["signed_bias_a"]
        b_b = stats["signed_bias_b"]

        lam_star_convex_theory = TheoryValidator.derive_lambda_star_convex(
            bias_sq_rf=stats["bias_a"],
            var_rf=stats["var_a"],
            bias_sq_xgb=stats["bias_b"],
            var_xgb=stats["var_b"],
            covariance=stats["covariance"],
            signed_bias_rf=b_a,
            signed_bias_xgb=b_b,
        )

        lam_star_prob_theory = TheoryValidator.derive_lambda_star_probabilistic(
            bias_sq_rf=stats["bias_a"],
            var_rf=stats["var_a"],
            bias_sq_xgb=stats["bias_b"],
            var_xgb=stats["var_b"],
            bias_diff_sq=stats["bias_diff_sq"],
            signed_bias_rf=b_a,
            signed_bias_xgb=b_b,
        )

        lam_star_convex_emp = TheoryValidator.find_empirical_lambda_star(convex_results)
        lam_star_prob_emp   = TheoryValidator.find_empirical_lambda_star(prob_results)

        # MSE at optimal λ
        mse_convex_star = convex_results[lam_star_convex_emp].mse
        mse_prob_star   = prob_results[lam_star_prob_emp].mse

        return TheoryValidationRow(
            condition_name=condition_name,
            pair_name=pair_name,
            bias_a=stats["bias_a"],
            var_a=stats["var_a"],
            bias_b=stats["bias_b"],
            var_b=stats["var_b"],
            covariance=stats["covariance"],
            rho=stats["rho"],
            lambda_star_convex_theory=lam_star_convex_theory,
            lambda_star_convex_empirical=lam_star_convex_emp,
            lambda_star_prob_theory=lam_star_prob_theory,
            lambda_star_prob_empirical=lam_star_prob_emp,
            mse_a=stats["mse_a"],
            mse_b=stats["mse_b"],
            mse_convex_at_star=mse_convex_star,
            mse_prob_at_star=mse_prob_star,
        )
