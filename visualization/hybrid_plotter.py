"""
visualization/hybrid_plotter.py — HybridPlotter.

Plot 5: Bias²-Variance space showing convex and probabilistic hybrid curves
        with RF and XGB endpoints, and λ* markers.

Plot 6: MSE vs λ for both hybrid types, with theoretical and empirical λ*
        vertical lines.
"""

from __future__ import annotations
import _path_setup

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

from config import COLORS, MODEL_LABELS, LAMBDA_VALUES
from pipeline_types import HybridResult, BiasVarianceResult


class HybridPlotter:
    """
    Generates Plots 5 and 6 for the hybrid analysis section.
    """

    def __init__(
        self,
        figsize: Tuple[float, float] = (9, 7),
        dpi: int = 150,
        style: str = "seaborn-v0_8-whitegrid",
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.style = style

    # ── Plot 5: Hybrid in Bias²-Variance space ────────────────────────────

    def plot_hybrid_bv_space(
        self,
        rf_bv: BiasVarianceResult,
        xgb_bv: BiasVarianceResult,
        convex_results: Dict[float, HybridResult],
        prob_results: Dict[float, HybridResult],
        lambda_star_convex: float,
        lambda_star_prob: float,
        title: str = "Plot 5 · Hybrid Curves in Bias²–Variance Space",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Draw the bias²-variance plane with:
          - RF endpoint (λ=0, blue dot)
          - XGB endpoint (λ=1, red dot)
          - Convex hybrid curve (solid green)
          - Probabilistic hybrid curve (dashed amber)
          - λ* markers on each curve
          - Iso-error contours
        """
        try:
            plt.style.use(self.style)
        except Exception:
            pass

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # ── Collect data ──────────────────────────────────────────────────
        lambdas_sorted = sorted(convex_results.keys())

        c_bias = [convex_results[l].bias_squared for l in lambdas_sorted]
        c_var  = [convex_results[l].variance     for l in lambdas_sorted]
        p_bias = [prob_results[l].bias_squared   for l in lambdas_sorted]
        p_var  = [prob_results[l].variance       for l in lambdas_sorted]

        all_bias = c_bias + p_bias + [rf_bv.bias_squared, xgb_bv.bias_squared]
        all_var  = c_var  + p_var  + [rf_bv.variance,     xgb_bv.variance]

        # ── Iso-error contours ────────────────────────────────────────────
        self._draw_iso_contours(ax, all_bias, all_var)

        # ── Convex curve ──────────────────────────────────────────────────
        ax.plot(
            c_bias, c_var,
            color=COLORS["convex"],
            linewidth=2.5,
            label="Convex hybrid",
            zorder=3,
            solid_capstyle="round",
        )

        # ── Probabilistic curve ───────────────────────────────────────────
        ax.plot(
            p_bias, p_var,
            color=COLORS["probabilistic"],
            linewidth=2.5,
            linestyle="--",
            label="Probabilistic hybrid",
            zorder=3,
        )

        # ── Annotate λ on curves every other point ────────────────────────
        for idx, lam in enumerate(lambdas_sorted):
            if idx % 2 == 0:  # Every other λ to avoid clutter
                ax.annotate(
                    f"λ={lam:.1f}",
                    (c_bias[idx], c_var[idx]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                    color=COLORS["convex"],
                    alpha=0.8,
                )

        # ── λ* markers ───────────────────────────────────────────────────
        c_star_result = convex_results.get(lambda_star_convex)
        p_star_result = prob_results.get(lambda_star_prob)

        if c_star_result:
            ax.scatter(
                [c_star_result.bias_squared], [c_star_result.variance],
                s=200, color=COLORS["convex"],
                marker="*", zorder=7, edgecolors="white", linewidths=0.8,
                label=f"λ*_convex = {lambda_star_convex:.2f}",
            )

        if p_star_result:
            ax.scatter(
                [p_star_result.bias_squared], [p_star_result.variance],
                s=200, color=COLORS["probabilistic"],
                marker="*", zorder=7, edgecolors="white", linewidths=0.8,
                label=f"λ*_prob = {lambda_star_prob:.2f}",
            )

        # ── Endpoints ─────────────────────────────────────────────────────
        ax.scatter(
            [rf_bv.bias_squared], [rf_bv.variance],
            s=180, color=COLORS["rf"],
            zorder=8, marker="D", edgecolors="white", linewidths=0.8,
            label=f"RF (λ=0)  MSE={rf_bv.mse:.2f}",
        )
        ax.scatter(
            [xgb_bv.bias_squared], [xgb_bv.variance],
            s=180, color=COLORS["xgb"],
            zorder=8, marker="D", edgecolors="white", linewidths=0.8,
            label=f"XGB (λ=1)  MSE={xgb_bv.mse:.2f}",
        )

        ax.annotate("RF\n(λ=0)",  (rf_bv.bias_squared,  rf_bv.variance),
                    textcoords="offset points", xytext=(-28, 6), fontsize=9,
                    color=COLORS["rf"], fontweight="bold")
        ax.annotate("XGB\n(λ=1)", (xgb_bv.bias_squared, xgb_bv.variance),
                    textcoords="offset points", xytext=(6, 6), fontsize=9,
                    color=COLORS["xgb"], fontweight="bold")

        # ── Labels ────────────────────────────────────────────────────────
        ax.set_xlabel("Bias²", fontsize=13, labelpad=8)
        ax.set_ylabel("Variance", fontsize=13, labelpad=8)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

        self._pad_axes(ax, all_bias, all_var)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ── Plot 6: MSE vs λ ──────────────────────────────────────────────────

    def plot_mse_vs_lambda(
        self,
        convex_results: Dict[float, HybridResult],
        prob_results: Dict[float, HybridResult],
        lambda_star_convex_theory: float,
        lambda_star_prob_theory: float,
        lambda_star_convex_empirical: Optional[float] = None,
        lambda_star_prob_empirical: Optional[float] = None,
        rf_mse: Optional[float] = None,
        xgb_mse: Optional[float] = None,
        title: str = "Plot 6 · MSE vs λ",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Draw MSE as a function of λ for both hybrid types.

        Vertical lines: theoretical λ* (dashed) and empirical λ* (dotted).
        Dots: empirical optimal on the curve.
        Horizontal lines: RF and XGB baseline MSE (light, for reference).
        """
        try:
            plt.style.use(self.style)
        except Exception:
            pass

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        lambdas = sorted(convex_results.keys())
        c_mse = [convex_results[l].mse for l in lambdas]
        p_mse = [prob_results[l].mse   for l in lambdas]

        # ── MSE curves ────────────────────────────────────────────────────
        ax.plot(lambdas, c_mse,
                color=COLORS["convex"],    linewidth=2.5,
                label="Convex hybrid", zorder=3)
        ax.plot(lambdas, p_mse,
                color=COLORS["probabilistic"], linewidth=2.5,
                linestyle="--",
                label="Probabilistic hybrid", zorder=3)

        # ── Baseline MSEs (horizontal reference lines) ────────────────────
        if rf_mse is not None:
            ax.axhline(rf_mse,  color=COLORS["rf"],  linewidth=1.0,
                       linestyle=":", alpha=0.7, label=f"RF MSE = {rf_mse:.2f}")
        if xgb_mse is not None:
            ax.axhline(xgb_mse, color=COLORS["xgb"], linewidth=1.0,
                       linestyle=":", alpha=0.7, label=f"XGB MSE = {xgb_mse:.2f}")

        # ── Empirical dots at curve minima ────────────────────────────────
        if lambda_star_convex_empirical is not None:
            mse_c_star = convex_results[lambda_star_convex_empirical].mse
            ax.scatter(
                [lambda_star_convex_empirical], [mse_c_star],
                s=120, color=COLORS["convex"],
                zorder=7, marker="o", edgecolors="white", linewidths=1,
                label=f"Empirical λ*_convex = {lambda_star_convex_empirical:.2f}",
            )

        if lambda_star_prob_empirical is not None:
            mse_p_star = prob_results[lambda_star_prob_empirical].mse
            ax.scatter(
                [lambda_star_prob_empirical], [mse_p_star],
                s=120, color=COLORS["probabilistic"],
                zorder=7, marker="o", edgecolors="white", linewidths=1,
                label=f"Empirical λ*_prob = {lambda_star_prob_empirical:.2f}",
            )

        # ── Theoretical λ* vertical lines ────────────────────────────────
        ax.axvline(
            lambda_star_convex_theory,
            color=COLORS["convex"],
            linewidth=1.5, linestyle="--", alpha=0.7,
            label=f"Theory λ*_convex = {lambda_star_convex_theory:.2f}",
        )
        ax.axvline(
            lambda_star_prob_theory,
            color=COLORS["probabilistic"],
            linewidth=1.5, linestyle="--", alpha=0.7,
            label=f"Theory λ*_prob = {lambda_star_prob_theory:.2f}",
        )

        # ── Labels ────────────────────────────────────────────────────────
        ax.set_xlabel("λ  (0 = pure RF, 1 = pure XGB)", fontsize=13, labelpad=8)
        ax.set_ylabel("MSE (Bias² + Variance)", fontsize=13, labelpad=8)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlim(-0.03, 1.03)
        ax.legend(loc="upper center", fontsize=8.5, framealpha=0.92,
                  ncol=2, bbox_to_anchor=(0.5, -0.12))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ── Plot 7: Correlation bar chart across pairs ────────────────────────

    def plot_correlation_barchart(
        self,
        pair_rhos: Dict[str, float],
        pair_labels: Optional[Dict[str, str]] = None,
        save_path: Optional[str] = None,
        title: str = "Plot 7 · Prediction Correlation (ρ) Across Hybrid Pairs",
    ) -> plt.Figure:
        """
        Grouped bar chart: one bar per pair, coloured by COLORS[pair_name].
        Each bar is annotated with its ρ value. A dashed reference line at
        ρ=0.9 marks the typical within-family correlation.
        """
        try:
            plt.style.use(self.style)
        except Exception:
            pass

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        pair_labels = pair_labels or {p: p for p in pair_rhos}
        names  = list(pair_rhos.keys())
        labels = [pair_labels.get(n, n) for n in names]
        rhos   = [pair_rhos[n] for n in names]
        colors = [COLORS.get(n, "#64748B") for n in names]

        x = np.arange(len(names))
        bars = ax.bar(x, rhos, color=colors, edgecolor="white", linewidth=1.2, zorder=3)

        # Annotate each bar with its numeric ρ value
        for rect, val in zip(bars, rhos):
            ax.annotate(
                f"{val:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
            )

        # Reference line for typical within-family correlation
        ax.axhline(
            0.9, color="#94A3B8", linestyle="--", linewidth=1.0, alpha=0.8,
            zorder=2, label="ρ = 0.9 (typical within-family)",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("ρ(A, B)", fontsize=12, labelpad=8)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)

        y_top = max(1.02, max(rhos) + 0.08)
        y_bot = min(0.0, min(rhos) - 0.08)
        ax.set_ylim(y_bot, y_top)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.92)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ── Plot 8: Multi-pair MSE vs λ ───────────────────────────────────────

    def plot_mse_vs_lambda_multipair(
        self,
        pairs_convex: Dict[str, Dict[float, HybridResult]],
        pair_labels: Optional[Dict[str, str]] = None,
        save_path: Optional[str] = None,
        title: str = "Plot 8 · MSE vs λ — Convex Hybrid (All Pairs)",
    ) -> plt.Figure:
        """
        One MSE-vs-λ line per hybrid pair (convex only). Empirical λ* for
        each pair is marked with a vertical dashed line in the same colour.
        """
        try:
            plt.style.use(self.style)
        except Exception:
            pass

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        pair_labels = pair_labels or {p: p for p in pairs_convex}

        for pair_name, convex in pairs_convex.items():
            if not convex:
                continue
            lambdas = sorted(convex.keys())
            mses    = [convex[l].mse for l in lambdas]
            color   = COLORS.get(pair_name, "#64748B")
            label   = pair_labels.get(pair_name, pair_name)

            ax.plot(
                lambdas, mses,
                color=color, linewidth=2.2,
                label=label, zorder=3,
            )

            # Empirical λ* for this pair
            lam_star = min(convex, key=lambda l: convex[l].mse)
            ax.axvline(
                lam_star, color=color,
                linewidth=1.2, linestyle="--", alpha=0.6, zorder=2,
            )
            ax.scatter(
                [lam_star], [convex[lam_star].mse],
                s=100, color=color, marker="o",
                edgecolors="white", linewidths=1, zorder=5,
            )

        ax.set_xlabel("λ  (0 = pure A, 1 = pure B)", fontsize=12, labelpad=8)
        ax.set_ylabel("MSE (Bias² + Variance)", fontsize=12, labelpad=8)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlim(-0.03, 1.03)
        ax.legend(loc="best", fontsize=9, framealpha=0.92)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ── Helpers ───────────────────────────────────────────────────────────

    def _draw_iso_contours(
        self,
        ax: plt.Axes,
        all_bias: List[float],
        all_var: List[float],
        n_contours: int = 5,
    ):
        pad_b = (max(all_bias) - min(all_bias)) * 0.2 + 1e-6
        pad_v = (max(all_var)  - min(all_var))  * 0.2 + 1e-6

        b_grid = np.linspace(max(0, min(all_bias) - pad_b), max(all_bias) + pad_b, 300)
        v_grid = np.linspace(max(0, min(all_var)  - pad_v), max(all_var)  + pad_v, 300)
        B, V = np.meshgrid(b_grid, v_grid)
        MSE = B + V

        all_mse = [b + v for b, v in zip(all_bias, all_var)]
        levels = np.linspace(min(all_mse), max(all_mse), n_contours + 2)[1:-1]

        cs = ax.contour(B, V, MSE,
                        levels=levels,
                        colors=[COLORS["iso_contour"]],
                        linestyles="--",
                        linewidths=0.7,
                        alpha=0.5,
                        zorder=1)
        ax.clabel(cs, fmt="%.1f", fontsize=7, colors=COLORS["iso_contour"])

    def _pad_axes(
        self,
        ax: plt.Axes,
        all_bias: List[float],
        all_var: List[float],
        pad_frac: float = 0.12,
    ):
        b_range = max(all_bias) - min(all_bias) + 1e-6
        v_range = max(all_var)  - min(all_var)  + 1e-6
        ax.set_xlim(max(0, min(all_bias) - b_range * pad_frac),
                    max(all_bias) + b_range * pad_frac)
        ax.set_ylim(max(0, min(all_var)  - v_range * pad_frac),
                    max(all_var)  + v_range * pad_frac)
