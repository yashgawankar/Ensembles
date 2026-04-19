"""
analysis/metrics.py — MetricsComputer and TableBuilder.

MetricsComputer:  Derived metrics, trajectory analysis, hypothesis evaluation.
TableBuilder:     Assembles theory validation and hypothesis verdict tables as DataFrames.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline_types import (
    ConfigResult,
    ExperimentResult,
    HybridResult,
    HypothesisVerdict,
    TheoryValidationRow,
)
from analysis.results_store import ResultsStore
from experiments.theory_validator import TheoryValidator


# ─────────────────────────────────────────────
# MetricsComputer
# ─────────────────────────────────────────────

class MetricsComputer:
    """
    Utility class for derived metrics that go beyond raw bias²/variance.
    """

    @staticmethod
    def compute_mse(bias_squared: float, variance: float) -> float:
        return bias_squared + variance

    @staticmethod
    def trajectory_summary(experiment: ExperimentResult) -> pd.DataFrame:
        """
        For a single experiment, return a DataFrame summarising trajectories:

        Columns: model_type, parameter_value, bias_squared, variance, mse,
                 pareto_optimal (bool)
        """
        rows = []
        for cr in experiment.config_results:
            rows.append({
                "model_type":      cr.model_type,
                "parameter_value": cr.parameter_value,
                "bias_squared":    cr.bias_squared,
                "variance":        cr.variance,
                "mse":             cr.mse,
            })
        df = pd.DataFrame(rows)

        # Pareto optimality: a point is Pareto-dominated if there exists another
        # point with strictly lower bias² AND strictly lower variance.
        pareto_flags = []
        for _, row in df.iterrows():
            dominated = any(
                (other["bias_squared"] < row["bias_squared"] and
                 other["variance"]     < row["variance"])
                for _, other in df.iterrows()
            )
            pareto_flags.append(not dominated)
        df["pareto_optimal"] = pareto_flags

        return df.sort_values(["model_type", "parameter_value"])

    @staticmethod
    def compare_models_at_value(
        experiment: ExperimentResult,
        parameter_value,
    ) -> Dict[str, Dict]:
        """
        Compare all models at a specific parameter value.
        Returns {model_type: {bias_squared, variance, mse}}.
        """
        result = {}
        for cr in experiment.config_results:
            if cr.parameter_value == parameter_value:
                result[cr.model_type] = {
                    "bias_squared": cr.bias_squared,
                    "variance":     cr.variance,
                    "mse":          cr.mse,
                }
        return result

    @staticmethod
    def best_config_per_model(experiment: ExperimentResult) -> Dict[str, ConfigResult]:
        """Return the ConfigResult with minimum MSE for each model type."""
        best: Dict[str, ConfigResult] = {}
        for cr in experiment.config_results:
            if cr.model_type not in best or cr.mse < best[cr.model_type].mse:
                best[cr.model_type] = cr
        return best

    @staticmethod
    def sensitivity(experiment: ExperimentResult, model_type: str) -> float:
        """
        Sensitivity = std(MSE across parameter values) for one model.
        Higher sensitivity → model is more affected by this parameter.
        """
        mses = [
            cr.mse for cr in experiment.config_results
            if cr.model_type == model_type
        ]
        return float(np.std(mses)) if mses else 0.0

    @staticmethod
    def compute_iso_contour_values(
        bias_sq_list: List[float],
        variance_list: List[float],
        n_contours: int = 5,
    ) -> List[float]:
        """
        Compute MSE levels for iso-error contours on the bias²-variance plot.
        Returns n_contours evenly spaced MSE values between min and max observed.
        """
        mses = [b + v for b, v in zip(bias_sq_list, variance_list)]
        min_mse = min(mses)
        max_mse = max(mses)
        return list(np.linspace(min_mse, max_mse, n_contours))


# ─────────────────────────────────────────────
# TableBuilder
# ─────────────────────────────────────────────

class TableBuilder:
    """
    Assembles the two key tables for the report:
    1. Theory validation table  (Section 7c of execution plan)
    2. Hypothesis verdict table (Section 7d of execution plan)
    """

    def __init__(
        self,
        results_store: ResultsStore,
        theory_validator: Optional[TheoryValidator] = None,
    ):
        self.results_store = results_store
        self.theory_validator = theory_validator or TheoryValidator

    # ── Theory validation ─────────────────────────────────────────────────

    def build_theory_validation_table(self) -> pd.DataFrame:
        """
        Build the full theory validation table from stored TheoryValidationRows.
        """
        return self.results_store.theory_rows_to_dataframe()

    # ── Hypothesis verdicts ───────────────────────────────────────────────

    def build_hypothesis_verdict_table(
        self,
        experiments: Dict[str, ExperimentResult],
        hybrid_convex: Optional[Dict[float, HybridResult]] = None,
        hybrid_prob:   Optional[Dict[float, HybridResult]] = None,
        theory_rows:   Optional[List[TheoryValidationRow]] = None,
    ) -> pd.DataFrame:
        """
        Programmatically evaluate hypotheses H1-H7 where possible.
        Observed values are extracted from stored results; verdicts that
        require visual inspection are marked as 'Visual Required'.

        Returns a DataFrame with columns:
        Hypothesis, Description, Prediction, Observed, Verdict
        """
        hypotheses: List[HypothesisVerdict] = []

        # H1: Trajectory divergence
        h1_pred = (
            "Bagging trajectory moves DOWN (variance↓, bias² stable). "
            "Boosting trajectory moves LEFT then UP (bias↓, variance↑)."
        )
        h1_obs = self._evaluate_h1_trajectory(experiments)
        hypotheses.append(HypothesisVerdict(
            hypothesis_id="H1",
            description="Trajectory divergence in Bias²-Variance space",
            prediction=h1_pred,
            observed=h1_obs,
            verdict="Visual Required",
        ))

        # H2: Complexity effect – boosting more sensitive to depth
        h2_obs, h2_verdict = self._evaluate_h2_depth_sensitivity(experiments)
        hypotheses.append(HypothesisVerdict(
            hypothesis_id="H2",
            description="Boosting more sensitive to base learner depth than bagging",
            prediction="XGBoost MSE std across depth values > RF MSE std across depth values",
            observed=h2_obs,
            verdict=h2_verdict,
        ))

        # H3: Data scaling – boosting benefits more from larger training sets
        h3_obs, h3_verdict = self._evaluate_h3_data_scaling(experiments)
        hypotheses.append(HypothesisVerdict(
            hypothesis_id="H3",
            description="Boosting benefits more from additional data",
            prediction="XGBoost MSE improvement from n=500→8000 > RF improvement",
            observed=h3_obs,
            verdict=h3_verdict,
        ))

        # H4: Noise fragility – boosting degrades more under noise
        h4_obs, h4_verdict = self._evaluate_h4_noise(experiments)
        hypotheses.append(HypothesisVerdict(
            hypothesis_id="H4",
            description="Boosting degrades more under high noise",
            prediction="XGBoost variance increase under noise > RF variance increase",
            observed=h4_obs,
            verdict=h4_verdict,
        ))

        # H5: Hybrid dominance – optimal λ* beats both parents
        h5_obs, h5_verdict = self._evaluate_h5_hybrid_dominance(
            hybrid_convex, theory_rows
        )
        hypotheses.append(HypothesisVerdict(
            hypothesis_id="H5",
            description="Optimal λ* achieves lower MSE than either parent model",
            prediction="MSE(f_λ*) < min(MSE_RF, MSE_XGB)",
            observed=h5_obs,
            verdict=h5_verdict,
        ))

        # H6: Theoretical λ* accuracy
        h6_obs, h6_verdict = self._evaluate_h6_theory_accuracy(theory_rows)
        hypotheses.append(HypothesisVerdict(
            hypothesis_id="H6",
            description="Theoretical λ* matches empirical λ* within 0.1",
            prediction="|λ*_theory - λ*_empirical| ≤ 0.1 for convex hybrid",
            observed=h6_obs,
            verdict=h6_verdict,
        ))

        # H7: Convex achieves lower error than probabilistic
        h7_obs, h7_verdict = self._evaluate_h7_convex_vs_prob(
            hybrid_convex, hybrid_prob
        )
        hypotheses.append(HypothesisVerdict(
            hypothesis_id="H7",
            description="Convex hybrid achieves lower minimum MSE than probabilistic",
            prediction="min MSE(convex) < min MSE(probabilistic)",
            observed=h7_obs,
            verdict=h7_verdict,
        ))

        return pd.DataFrame([{
            "Hypothesis": h.hypothesis_id,
            "Description": h.description,
            "Prediction": h.prediction,
            "Observed": h.observed,
            "Verdict": h.verdict,
        } for h in hypotheses])

    # ── H-evaluation helpers ──────────────────────────────────────────────

    def _evaluate_h1_trajectory(self, experiments: Dict) -> str:
        exp = experiments.get("ensemble_size")
        if exp is None:
            return "Insufficient data"
        best = MetricsComputer.best_config_per_model(exp)
        summary_parts = []
        for mt, cr in best.items():
            summary_parts.append(f"{mt}: min MSE={cr.mse:.2f} at {cr.parameter_value} estimators")
        return " | ".join(summary_parts) + "  [see Plot 1 for trajectory shapes]"

    def _evaluate_h2_depth_sensitivity(self, experiments: Dict) -> Tuple[str, str]:
        exp = experiments.get("depth")
        if exp is None:
            return "No depth experiment found", "Not Evaluated"
        s_xgb = MetricsComputer.sensitivity(exp, "xgb")
        s_rf  = MetricsComputer.sensitivity(exp, "rf")
        obs = f"XGBoost MSE std across depths = {s_xgb:.4f} | RF MSE std = {s_rf:.4f}"
        verdict = "Supported" if s_xgb > s_rf else "Not Supported"
        return obs, verdict

    def _evaluate_h3_data_scaling(self, experiments: Dict) -> Tuple[str, str]:
        exp = experiments.get("train_size")
        if exp is None:
            return "No train_size experiment found", "Not Evaluated"

        def get_mse(model, size):
            cr = exp.get(model, size)
            return cr.mse if cr else None

        all_sizes = sorted({
            cr.parameter_value for cr in exp.config_results
            if isinstance(cr.parameter_value, int)
        })
        if len(all_sizes) < 2:
            return "Insufficient data points", "Not Evaluated"

        n_small, n_large = all_sizes[0], all_sizes[-1]
        mse_xgb_small = get_mse("xgb", n_small)
        mse_xgb_large = get_mse("xgb", n_large)
        mse_rf_small  = get_mse("rf",  n_small)
        mse_rf_large  = get_mse("rf",  n_large)

        if any(v is None for v in [mse_xgb_small, mse_xgb_large, mse_rf_small, mse_rf_large]):
            return "Missing data", "Not Evaluated"

        delta_xgb = mse_xgb_small - mse_xgb_large
        delta_rf  = mse_rf_small  - mse_rf_large
        obs = (
            f"XGBoost MSE drop (n={n_small}→{n_large}): {delta_xgb:.4f} | "
            f"RF MSE drop: {delta_rf:.4f}"
        )
        verdict = "Supported" if delta_xgb > delta_rf else "Not Supported"
        return obs, verdict

    def _evaluate_h4_noise(self, experiments: Dict) -> Tuple[str, str]:
        exp = experiments.get("noise")
        if exp is None:
            return "No noise experiment found", "Not Evaluated"

        sigmas = sorted({
            cr.parameter_value for cr in exp.config_results
            if isinstance(cr.parameter_value, (int, float))
        })
        if len(sigmas) < 2:
            return "Insufficient data", "Not Evaluated"

        low_s, high_s = sigmas[0], sigmas[-1]

        cr_xgb_low  = exp.get("xgb", low_s)
        cr_xgb_high = exp.get("xgb", high_s)
        cr_rf_low   = exp.get("rf",  low_s)
        cr_rf_high  = exp.get("rf",  high_s)

        if any(cr is None for cr in [cr_xgb_low, cr_xgb_high, cr_rf_low, cr_rf_high]):
            return "Missing data", "Not Evaluated"

        delta_var_xgb = cr_xgb_high.variance - cr_xgb_low.variance
        delta_var_rf  = cr_rf_high.variance   - cr_rf_low.variance
        obs = (
            f"XGBoost variance increase (σ={low_s}→{high_s}): {delta_var_xgb:.4f} | "
            f"RF: {delta_var_rf:.4f}"
        )
        verdict = "Supported" if delta_var_xgb > delta_var_rf else "Not Supported"
        return obs, verdict

    def _evaluate_h5_hybrid_dominance(
        self,
        hybrid_convex: Optional[Dict],
        theory_rows: Optional[List[TheoryValidationRow]],
    ) -> Tuple[str, str]:
        if hybrid_convex is None or not theory_rows:
            return "Hybrid results not available", "Not Evaluated"
        row = next((r for r in theory_rows if r.condition_name == "baseline"), None)
        if row is None:
            return "No baseline theory row", "Not Evaluated"

        min_parent = min(row.mse_rf, row.mse_xgb)
        mse_hybrid = row.mse_convex_at_star
        obs = (
            f"MSE RF={row.mse_rf:.4f}, MSE XGB={row.mse_xgb:.4f}, "
            f"MSE Convex@λ*={mse_hybrid:.4f}"
        )
        verdict = "Supported" if mse_hybrid < min_parent else "Not Supported"
        return obs, verdict

    def _evaluate_h6_theory_accuracy(
        self, theory_rows: Optional[List[TheoryValidationRow]]
    ) -> Tuple[str, str]:
        if not theory_rows:
            return "Theory rows not computed", "Not Evaluated"
        row = next((r for r in theory_rows if r.condition_name == "baseline"), None)
        if row is None:
            return "No baseline row", "Not Evaluated"
        diff = abs(row.lambda_star_convex_theory - row.lambda_star_convex_empirical)
        obs = (
            f"λ*_theory={row.lambda_star_convex_theory:.4f}, "
            f"λ*_empirical={row.lambda_star_convex_empirical:.4f}, "
            f"|diff|={diff:.4f}"
        )
        verdict = "Supported" if diff <= 0.1 else "Not Supported"
        return obs, verdict

    def _evaluate_h7_convex_vs_prob(
        self,
        hybrid_convex: Optional[Dict],
        hybrid_prob:   Optional[Dict],
    ) -> Tuple[str, str]:
        if hybrid_convex is None or hybrid_prob is None:
            return "Hybrid results not available", "Not Evaluated"
        min_mse_convex = min(hr.mse for hr in hybrid_convex.values())
        min_mse_prob   = min(hr.mse for hr in hybrid_prob.values())
        obs = (
            f"Min MSE convex={min_mse_convex:.4f}, "
            f"Min MSE probabilistic={min_mse_prob:.4f}"
        )
        verdict = "Supported" if min_mse_convex < min_mse_prob else "Not Supported"
        return obs, verdict
