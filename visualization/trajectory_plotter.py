"""
visualization/trajectory_plotter.py — TrajectoryPlotter.

Generates Plots 1-4: Bias²-Variance trajectory atlas.

Design principles:
  - Each model (RF, XGB, Bagged-XGB) is a distinct trajectory in bias²-variance space
  - Arrows show direction of increasing parameter value
  - Iso-error contours (dashed gray) show MSE levels
  - Points labelled with the parameter value
  - Publication-quality but readable on screen
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

from config import COLORS, MODEL_LABELS, EXPERIMENT_VARIATIONS
from pipeline_types import ExperimentResult
from analysis.results_store import ResultsStore
from analysis.metrics import MetricsComputer


class TrajectoryPlotter:
    """
    Produces the four trajectory plots for experiments E1-E4.
    """

    def __init__(
        self,
        results_store: ResultsStore,
        figsize: Tuple[float, float] = (8, 7),
        dpi: int = 150,
        style: str = "seaborn-v0_8-whitegrid",
    ):
        self.results_store = results_store
        self.figsize = figsize
        self.dpi = dpi
        self.style = style

    # ── Public plot generators ────────────────────────────────────────────

    def plot_ensemble_size(self, save_path: Optional[str] = None) -> plt.Figure:
        exp = self.results_store.get_experiment_result("ensemble_size")
        return self._plot_trajectory(
            experiment=exp,
            title="E1 · Ensemble Size Trajectory",
            param_label="n_estimators",
            save_path=save_path,
        )

    def plot_depth(self, save_path: Optional[str] = None) -> plt.Figure:
        exp = self.results_store.get_experiment_result("depth")
        return self._plot_trajectory(
            experiment=exp,
            title="E2 · Base Learner Depth Trajectory",
            param_label="max_depth",
            save_path=save_path,
        )

    def plot_noise(self, save_path: Optional[str] = None) -> plt.Figure:
        exp = self.results_store.get_experiment_result("noise")
        return self._plot_trajectory(
            experiment=exp,
            title="E3 · Noise Level Trajectory",
            param_label="σ",
            save_path=save_path,
        )

    def plot_train_size(self, save_path: Optional[str] = None) -> plt.Figure:
        exp = self.results_store.get_experiment_result("train_size")
        return self._plot_trajectory(
            experiment=exp,
            title="E4 · Training Set Size Trajectory",
            param_label="n_train",
            save_path=save_path,
        )

    def plot_all(self, save_dir: Optional[str] = None) -> List[plt.Figure]:
        """Generate all four trajectory plots."""
        plots = []
        for name, method in [
            ("ensemble_size", self.plot_ensemble_size),
            ("depth",         self.plot_depth),
            ("noise",         self.plot_noise),
            ("train_size",    self.plot_train_size),
        ]:
            save_path = f"{save_dir}/{name}.png" if save_dir else None
            fig = method(save_path=save_path)
            plots.append(fig)
        return plots

    # ── Core plotting logic ───────────────────────────────────────────────

    def _plot_trajectory(
        self,
        experiment: ExperimentResult,
        title: str,
        param_label: str,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generic trajectory plot in bias²-variance space for one experiment.
        """
        try:
            plt.style.use(self.style)
        except Exception:
            pass

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Gather all bias² and variance values across all models for axis scaling
        all_bias  = [cr.bias_squared for cr in experiment.config_results]
        all_var   = [cr.variance     for cr in experiment.config_results]

        # ── Iso-error contours ────────────────────────────────────────────
        self._draw_iso_contours(ax, all_bias, all_var)

        # ── Per-model trajectories ────────────────────────────────────────
        for model_type in ["rf", "xgb", "bagged_xgb"]:
            param_vals, bias_sqs, variances = experiment.get_model_trajectory(model_type)
            if not param_vals:
                continue

            color = COLORS[model_type]
            label = MODEL_LABELS[model_type]

            # Sort by parameter value (handle None for RF unlimited depth)
            pairs = sorted(
                zip(param_vals, bias_sqs, variances),
                key=lambda t: (t[0] is None, t[0])
            )
            param_vals_sorted = [p[0] for p in pairs]
            bias_sorted       = [p[1] for p in pairs]
            var_sorted        = [p[2] for p in pairs]

            # Trajectory line
            ax.plot(
                bias_sorted, var_sorted,
                color=color,
                linewidth=2.0,
                alpha=0.75,
                zorder=3,
            )

            # Points
            ax.scatter(
                bias_sorted, var_sorted,
                color=color,
                s=80,
                zorder=5,
                edgecolors="white",
                linewidths=0.8,
                label=label,
            )

            # Labels on points
            for pv, bsq, v in zip(param_vals_sorted, bias_sorted, var_sorted):
                lbl = "∞" if pv is None else str(pv)
                ax.annotate(
                    lbl,
                    (bsq, v),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=7.5,
                    color=color,
                    fontweight="semibold",
                    zorder=6,
                )

            # Direction arrows (between consecutive points)
            self._draw_arrows(ax, bias_sorted, var_sorted, color)

        # ── Axes and labels ───────────────────────────────────────────────
        ax.set_xlabel("Bias²", fontsize=13, labelpad=8)
        ax.set_ylabel("Variance", fontsize=13, labelpad=8)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)

        # Add parameter label to corner
        ax.text(
            0.98, 0.02, f"x = {param_label}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9, color="#555",
            style="italic",
        )

        # Legend (custom handles to match model colours)
        handles = [
            Line2D([0], [0], color=COLORS[m], linewidth=2, marker="o",
                   markersize=6, label=MODEL_LABELS[m])
            for m in ["rf", "xgb", "bagged_xgb"]
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.9)

        # Padding
        self._pad_axes(ax, all_bias, all_var)

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
        """Draw dashed iso-MSE contours in the background."""
        pad_b = (max(all_bias) - min(all_bias)) * 0.2 + 1e-6
        pad_v = (max(all_var)  - min(all_var))  * 0.2 + 1e-6

        b_min = max(0, min(all_bias) - pad_b)
        b_max = max(all_bias) + pad_b
        v_min = max(0, min(all_var) - pad_v)
        v_max = max(all_var) + pad_v

        b_grid = np.linspace(b_min, b_max, 400)
        v_grid = np.linspace(v_min, v_max, 400)
        B, V = np.meshgrid(b_grid, v_grid)
        MSE = B + V

        all_mse = [b + v for b, v in zip(all_bias, all_var)]
        mse_min, mse_max = min(all_mse), max(all_mse)
        levels = np.linspace(mse_min, mse_max, n_contours + 2)[1:-1]

        cs = ax.contour(
            B, V, MSE,
            levels=levels,
            colors=[COLORS["iso_contour"]],
            linestyles="--",
            linewidths=0.8,
            alpha=0.6,
            zorder=1,
        )
        ax.clabel(cs, fmt="%.1f", fontsize=7, colors=COLORS["iso_contour"])

    def _draw_arrows(
        self,
        ax: plt.Axes,
        bias_list: List[float],
        var_list: List[float],
        color: str,
    ):
        """Draw direction arrows between consecutive trajectory points."""
        for i in range(len(bias_list) - 1):
            dx = bias_list[i + 1] - bias_list[i]
            dy = var_list[i + 1]  - var_list[i]
            if abs(dx) + abs(dy) < 1e-12:
                continue
            # Midpoint arrow
            mx = (bias_list[i] + bias_list[i + 1]) / 2
            my = (var_list[i]  + var_list[i + 1])  / 2
            ax.annotate(
                "",
                xy=(mx + dx * 0.01, my + dy * 0.01),
                xytext=(mx - dx * 0.01, my - dy * 0.01),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.2,
                ),
                zorder=4,
            )

    def _pad_axes(
        self,
        ax: plt.Axes,
        all_bias: List[float],
        all_var: List[float],
        pad_frac: float = 0.15,
    ):
        """Add padding around the data range."""
        b_range = max(all_bias) - min(all_bias)
        v_range = max(all_var)  - min(all_var)
        b_pad = b_range * pad_frac + 1e-6
        v_pad = v_range * pad_frac + 1e-6

        ax.set_xlim(max(0, min(all_bias) - b_pad), max(all_bias) + b_pad)
        ax.set_ylim(max(0, min(all_var)  - v_pad), max(all_var)  + v_pad)
