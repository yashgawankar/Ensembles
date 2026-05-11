"""
analysis/metrics.py — MetricsComputer and TableBuilder.

MetricsComputer:  Derived metrics, trajectory analysis, hypothesis evaluation.
TableBuilder:     Assembles theory validation and hypothesis verdict tables as DataFrames.
"""

from __future__ import annotations
import _path_setup

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
        hybrid_pairs: Optional[Dict[str, Dict]] = None,
        theory_rows:  Optional[List[TheoryValidationRow]] = None,
    ) -> pd.DataFrame:
        """
        Programmatically evaluate hypotheses H1-H9 where possible.
        Observed values are extracted from stored results; verdicts that
        require visual inspection are marked as 'Visual Required'.

        Parameters
        ----------
        experiments  : dict of ExperimentResult per variation
        hybrid_pairs : dict keyed by pair name (e.g. "rf_xgb"), each value being
                       {"convex": {λ→HybridResult}, "prob": {λ→HybridResult},
                        "stats": dict, "preds_a": ndarray, "preds_b": ndarray}
        theory_rows  : list of TheoryValidationRow

        Returns a DataFrame with columns:
        Hypothesis, Description, Prediction, Observed, Verdict
        """
        hypotheses: List[HypothesisVerdict] = []
        hybrid_pairs = hybrid_pairs or {}

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

        # H5: Hybrid dominance – optimal λ* beats both parents (any pair)
        h5_obs, h5_verdict = self._evaluate_h5_hybrid_dominance(
            hybrid_pairs, theory_rows
        )
        hypotheses.append(HypothesisVerdict(
            hypothesis_id="H5",
            description="Optimal λ* achieves lower MSE than either parent model",
            prediction="MSE(f_λ*) < min(MSE_A, MSE_B) for at least one hybrid pair",
            observed=h5_obs,
            verdict=h5_verdict,
        ))

        # H6: Theoretical λ* accuracy (RF+XGB baseline)
        h6_obs, h6_verdict = self._evaluate_h6_theory_accuracy(theory_rows)
        hypotheses.append(HypothesisVerdict(
            hypothesis_id="H6",
            description="Theoretical λ* matches empirical λ* within 0.1",
            prediction="|λ*_theory - λ*_empirical| ≤ 0.1 for convex hybrid",
            observed=h6_obs,
            verdict=h6_verdict,
        ))

        # H7: Convex achieves lower error than probabilistic (per pair)
        h7_obs, h7_verdict = self._evaluate_h7_convex_vs_prob(hybrid_pairs)
        hypotheses.append(HypothesisVerdict(
            hypothesis_id="H7",
            description="Convex hybrid achieves lower minimum MSE than probabilistic",
            prediction="min MSE(convex) < min MSE(probabilistic) for each pair",
            observed=h7_obs,
            verdict=h7_verdict,
        ))

        # H8: Family diversity yields greater mixing benefit than strategy diversity
        h8_obs, h8_verdict = self._evaluate_h8_family_diversity(hybrid_pairs)
        hypotheses.append(HypothesisVerdict(
            hypothesis_id="H8",
            description=(
                "Cross-family pairs achieve larger MSE reduction at λ* than "
                "the within-family pair (RF+XGB)"
            ),
            prediction=(
                "mean MSE-reduction over {RF+SVR, XGB+SVR} > MSE-reduction for RF+XGB; "
                "mechanism: lower ρ(A,B) → wider variance-reduction channel"
            ),
            observed=h8_obs,
            verdict=h8_verdict,
        ))

        # H9: Convex advantage over probabilistic is amplified for cross-family pairs
        h9_obs, h9_verdict = self._evaluate_h9_convex_amplification(hybrid_pairs)
        hypotheses.append(HypothesisVerdict(
            hypothesis_id="H9",
            description=(
                "Convex hybrid advantage over probabilistic is amplified for "
                "cross-family pairs"
            ),
            prediction=(
                "(min_MSE_prob − min_MSE_convex) larger for {RF+SVR, XGB+SVR} "
                "than for RF+XGB; mechanism: larger D² = E[(μ_A − μ_B)²]"
            ),
            observed=h9_obs,
            verdict=h9_verdict,
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
        hybrid_pairs: Dict[str, Dict],
        theory_rows: Optional[List[TheoryValidationRow]],
    ) -> Tuple[str, str]:
        if not hybrid_pairs or not theory_rows:
            return "Hybrid results not available", "Not Evaluated"

        # For each pair, find the baseline row and compare MSE_convex@λ* with min(MSE_A, MSE_B).
        per_pair_obs = []
        any_supported = False
        for pair_name in hybrid_pairs.keys():
            row = next(
                (r for r in theory_rows
                 if r.condition_name == "baseline" and r.pair_name == pair_name),
                None,
            )
            if row is None:
                continue
            min_parent = min(row.mse_a, row.mse_b)
            reduction = min_parent - row.mse_convex_at_star
            if row.mse_convex_at_star < min_parent:
                any_supported = True
            per_pair_obs.append(
                f"{pair_name}: MSE_A={row.mse_a:.4f}, MSE_B={row.mse_b:.4f}, "
                f"MSE_Convex@λ*={row.mse_convex_at_star:.4f}, reduction={reduction:.4f}"
            )

        if not per_pair_obs:
            return "No baseline theory rows for any pair", "Not Evaluated"

        obs = " | ".join(per_pair_obs)
        verdict = "Supported" if any_supported else "Not Supported"
        return obs, verdict

    def _evaluate_h6_theory_accuracy(
        self, theory_rows: Optional[List[TheoryValidationRow]]
    ) -> Tuple[str, str]:
        if not theory_rows:
            return "Theory rows not computed", "Not Evaluated"
        # Use RF+XGB baseline if available, else first baseline row
        row = next(
            (r for r in theory_rows
             if r.condition_name == "baseline" and r.pair_name == "rf_xgb"),
            None,
        )
        if row is None:
            row = next((r for r in theory_rows if r.condition_name == "baseline"), None)
        if row is None:
            return "No baseline row", "Not Evaluated"
        diff = abs(row.lambda_star_convex_theory - row.lambda_star_convex_empirical)
        obs = (
            f"pair={row.pair_name}, λ*_theory={row.lambda_star_convex_theory:.4f}, "
            f"λ*_empirical={row.lambda_star_convex_empirical:.4f}, "
            f"|diff|={diff:.4f}"
        )
        verdict = "Supported" if diff <= 0.1 else "Not Supported"
        return obs, verdict

    def _evaluate_h7_convex_vs_prob(
        self,
        hybrid_pairs: Dict[str, Dict],
    ) -> Tuple[str, str]:
        if not hybrid_pairs:
            return "Hybrid results not available", "Not Evaluated"

        per_pair_obs = []
        all_supported = True
        any_evaluated = False
        for pair_name, payload in hybrid_pairs.items():
            convex = payload.get("convex")
            prob   = payload.get("prob")
            if not convex or not prob:
                continue
            any_evaluated = True
            min_mse_convex = min(hr.mse for hr in convex.values())
            min_mse_prob   = min(hr.mse for hr in prob.values())
            gap = min_mse_prob - min_mse_convex
            per_pair_obs.append(
                f"{pair_name}: convex={min_mse_convex:.4f}, prob={min_mse_prob:.4f}, "
                f"gap={gap:.4f}"
            )
            if min_mse_convex >= min_mse_prob:
                all_supported = False

        if not any_evaluated:
            return "No hybrid pairs available", "Not Evaluated"

        obs = " | ".join(per_pair_obs)
        verdict = "Supported" if all_supported else "Not Supported"
        return obs, verdict

    # ── H8: Family diversity ──────────────────────────────────────────────

    def _evaluate_h8_family_diversity(
        self,
        hybrid_pairs: Dict[str, Dict],
    ) -> Tuple[str, str]:
        """
        Compare MSE reduction at λ* for the within-family pair (rf_xgb)
        against the mean reduction across cross-family pairs (rf_svr, xgb_svr).
        Reduction = MSE_better_parent − min_MSE_convex.
        """
        if not hybrid_pairs:
            return "Hybrid results not available", "Not Evaluated"

        def reduction(payload: Dict) -> Optional[float]:
            convex = payload.get("convex")
            stats  = payload.get("stats")
            if not convex or not stats:
                return None
            min_mse_convex = min(hr.mse for hr in convex.values())
            min_parent = min(stats["mse_a"], stats["mse_b"])
            return min_parent - min_mse_convex

        within = reduction(hybrid_pairs.get("rf_xgb", {})) if "rf_xgb" in hybrid_pairs else None
        cross_reductions = []
        for cp in ("rf_svr", "xgb_svr"):
            r = reduction(hybrid_pairs.get(cp, {})) if cp in hybrid_pairs else None
            if r is not None:
                cross_reductions.append(r)

        if within is None or not cross_reductions:
            return "Need rf_xgb and at least one cross-family pair", "Not Evaluated"

        cross_mean = float(np.mean(cross_reductions))
        # ρ values for context
        rho_within = hybrid_pairs.get("rf_xgb", {}).get("stats", {}).get("rho")
        rho_cross  = [
            hybrid_pairs.get(p, {}).get("stats", {}).get("rho")
            for p in ("rf_svr", "xgb_svr") if p in hybrid_pairs
        ]
        rho_cross_mean = (
            float(np.mean([r for r in rho_cross if r is not None]))
            if rho_cross else None
        )

        obs = (
            f"reduction within(rf_xgb)={within:.4f} (ρ={rho_within:.3f}) | "
            f"mean cross={cross_mean:.4f}"
        )
        if rho_cross_mean is not None:
            obs += f" (mean ρ_cross={rho_cross_mean:.3f})"
        verdict = "Supported" if cross_mean > within else "Not Supported"
        return obs, verdict

    # ── H9: Convex advantage amplification ────────────────────────────────

    def _evaluate_h9_convex_amplification(
        self,
        hybrid_pairs: Dict[str, Dict],
    ) -> Tuple[str, str]:
        """
        For each pair, compute:
            convex_advantage = min_MSE_prob − min_MSE_convex
            D_sq             = stats["bias_diff_sq"]   (= E[(μ_A − μ_B)²])

        Verdict 'Supported' iff:
          (a) the within-family pair (rf_xgb) has the SMALLEST convex_advantage,
          (b) cross-family pairs have larger D² than within-family
              (mechanism check).
        """
        if not hybrid_pairs:
            return "Hybrid results not available", "Not Evaluated"

        per_pair = {}
        for pair_name, payload in hybrid_pairs.items():
            convex = payload.get("convex")
            prob   = payload.get("prob")
            stats  = payload.get("stats", {})
            if not convex or not prob or "bias_diff_sq" not in stats:
                continue
            min_mse_convex = min(hr.mse for hr in convex.values())
            min_mse_prob   = min(hr.mse for hr in prob.values())
            per_pair[pair_name] = {
                "advantage": min_mse_prob - min_mse_convex,
                "D_sq":       stats["bias_diff_sq"],
            }

        if "rf_xgb" not in per_pair or len(per_pair) < 2:
            return "Need rf_xgb plus at least one other pair", "Not Evaluated"

        within = per_pair["rf_xgb"]
        cross  = {p: v for p, v in per_pair.items() if p != "rf_xgb"}

        smallest_advantage = within["advantage"] <= min(v["advantage"] for v in cross.values())
        d_sq_check = all(v["D_sq"] > within["D_sq"] for v in cross.values())

        obs_parts = [
            f"{p}: adv={v['advantage']:.4f}, D²={v['D_sq']:.4f}"
            for p, v in per_pair.items()
        ]
        obs = " | ".join(obs_parts)
        verdict = "Supported" if (smallest_advantage and d_sq_check) else "Not Supported"
        return obs, verdict
