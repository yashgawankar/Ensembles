"""
experiments/hyperparameter_tuner.py — HyperparameterTuner.

Grid-searches model hyperparameters using a held-out validation set.
Tuning runs ONCE on the (train_pool, val) split and produces a single
tuned config per model. The bootstrap evaluator then uses these fixed
tuned configs across all M iterations — this preserves the bias-variance
methodology by separating model-selection variance from prediction variance.

XGB additionally uses early stopping on val MSE to choose n_estimators
within each grid-cell fit, so the search adapts ensemble depth automatically
without expanding the explicit grid over n_estimators.
"""

from __future__ import annotations
import _path_setup

import copy
import itertools
import time
from typing import Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("xgboost is required: pip install xgboost")


# ─────────────────────────────────────────────
# Default grids (medium size: 5-8 values per HP)
# ─────────────────────────────────────────────

DEFAULT_RF_GRID: Dict[str, List] = {
    "max_depth":         [3, 5, 8, 10, 15, None],
    "min_samples_leaf":  [1, 2, 5, 10],
}

DEFAULT_XGB_GRID: Dict[str, List] = {
    "learning_rate": [0.01, 0.03, 0.1, 0.3],
    "max_depth":     [3, 5, 7, 9],
    "subsample":     [0.7, 0.85, 1.0],
}

DEFAULT_SVR_GRID: Dict[str, List] = {
    "C":       [0.1, 1.0, 10.0, 100.0, 1000.0],
    "gamma":   ["scale", 0.001, 0.01, 0.1],
    "epsilon": [0.1, 1.0],
}


# ─────────────────────────────────────────────
# HyperparameterTuner
# ─────────────────────────────────────────────

class HyperparameterTuner:
    """
    Grid-search hyperparameters for RF, XGB, and SVR using a held-out
    validation set. Optimises val-set MSE.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.verbose = verbose

    # ── Public API ────────────────────────────────────────────────────────

    def tune_all(
        self,
        baseline_params: Dict[str, Dict],
        rf_grid: Optional[Dict] = None,
        xgb_grid: Optional[Dict] = None,
        svr_grid: Optional[Dict] = None,
        max_n_estimators: int = 500,
        early_stopping_rounds: int = 20,
    ) -> Dict[str, Dict]:
        """
        Tune all three models and return updated baseline configs.

        Returns
        -------
        Dict[str, Dict] — same keys as `baseline_params`. Tuned overrides
        applied to "rf", "xgb", "svr". The "bagged_xgb" entry inherits
        the tuned XGB inner config.
        """
        tuned: Dict[str, Dict] = copy.deepcopy(baseline_params)

        if "rf" in baseline_params:
            tuned["rf"] = self.tune_rf(baseline_params["rf"], grid=rf_grid)

        if "xgb" in baseline_params:
            tuned["xgb"] = self.tune_xgb(
                baseline_params["xgb"],
                grid=xgb_grid,
                max_n_estimators=max_n_estimators,
                early_stopping_rounds=early_stopping_rounds,
            )

        if "svr" in baseline_params:
            tuned["svr"] = self.tune_svr(baseline_params["svr"], grid=svr_grid)

        # Bagged-XGB inner: copy tuned XGB params (excluding n_jobs and seeds
        # already baked in) so the inner learner reflects the chosen config.
        if "bagged_xgb" in baseline_params and "xgb" in tuned:
            inner = copy.deepcopy(tuned["bagged_xgb"]["inner"])
            for hp in ("learning_rate", "max_depth", "subsample"):
                if hp in tuned["xgb"]:
                    inner[hp] = tuned["xgb"][hp]
            # Inherit tuned n_estimators too, but clip so outer×inner trees
            # remain tractable
            tuned_n = tuned["xgb"].get("n_estimators", inner.get("n_estimators"))
            inner["n_estimators"] = min(tuned_n, 100)
            tuned["bagged_xgb"]["inner"] = inner

        return tuned

    # ── Per-model tuners ──────────────────────────────────────────────────

    def tune_rf(
        self,
        base_config: Dict,
        grid: Optional[Dict] = None,
    ) -> Dict:
        grid = grid or DEFAULT_RF_GRID
        if self.verbose:
            n_combos = _grid_size(grid)
            print(f"  Tuning RF: {n_combos} grid combinations...")
        t0 = time.time()

        best = self._grid_search(
            base_config=base_config,
            grid=grid,
            estimator_factory=lambda cfg: RandomForestRegressor(**cfg),
        )
        if self.verbose:
            print(
                f"    RF best: val_mse={best['mse']:.4f}  "
                f"params={best['delta']}  ({time.time() - t0:.1f}s)"
            )
        return best["config"]

    def tune_xgb(
        self,
        base_config: Dict,
        grid: Optional[Dict] = None,
        max_n_estimators: int = 500,
        early_stopping_rounds: int = 20,
    ) -> Dict:
        """
        Grid-search XGB with val-set early stopping. n_estimators is chosen
        adaptively per grid-cell via best_iteration (capped at max_n_estimators).
        """
        grid = grid or DEFAULT_XGB_GRID
        if self.verbose:
            n_combos = _grid_size(grid)
            print(f"  Tuning XGB: {n_combos} grid combinations (early stopping)...")
        t0 = time.time()

        keys = list(grid.keys())
        values = [grid[k] for k in keys]

        best = {"mse": float("inf"), "config": None, "delta": None}

        for combo in itertools.product(*values):
            delta = dict(zip(keys, combo))
            cfg = copy.deepcopy(base_config)
            cfg.update(delta)
            cfg["n_estimators"] = max_n_estimators
            # XGBRegressor uses 'seed' or 'random_state' depending on version
            if "random_state" in cfg:
                cfg.setdefault("seed", cfg.pop("random_state"))

            chosen_n_est, mse = self._fit_xgb_with_early_stopping(
                cfg, early_stopping_rounds=early_stopping_rounds
            )

            if mse < best["mse"]:
                final_cfg = copy.deepcopy(base_config)
                final_cfg.update(delta)
                final_cfg["n_estimators"] = int(chosen_n_est)
                best = {"mse": mse, "config": final_cfg, "delta": delta,
                        "n_est": int(chosen_n_est)}

        if self.verbose:
            print(
                f"    XGB best: val_mse={best['mse']:.4f}  "
                f"params={best['delta']}  n_estimators={best['n_est']}  "
                f"({time.time() - t0:.1f}s)"
            )
        return best["config"]

    def tune_svr(
        self,
        base_config: Dict,
        grid: Optional[Dict] = None,
    ) -> Dict:
        grid = grid or DEFAULT_SVR_GRID
        if self.verbose:
            n_combos = _grid_size(grid)
            print(f"  Tuning SVR: {n_combos} grid combinations (slow; O(n²))...")
        t0 = time.time()

        best = self._grid_search(
            base_config=base_config,
            grid=grid,
            estimator_factory=lambda cfg: SVR(**cfg),
        )
        if self.verbose:
            print(
                f"    SVR best: val_mse={best['mse']:.4f}  "
                f"params={best['delta']}  ({time.time() - t0:.1f}s)"
            )
        return best["config"]

    # ── Internals ─────────────────────────────────────────────────────────

    def _grid_search(
        self,
        base_config: Dict,
        grid: Dict[str, List],
        estimator_factory,
    ) -> Dict:
        """Generic grid search returning {mse, config, delta}."""
        keys = list(grid.keys())
        values = [grid[k] for k in keys]

        best = {"mse": float("inf"), "config": None, "delta": None}

        for combo in itertools.product(*values):
            delta = dict(zip(keys, combo))
            cfg = copy.deepcopy(base_config)
            cfg.update(delta)

            try:
                est = estimator_factory(cfg)
                est.fit(self.X_train, self.y_train)
                preds = est.predict(self.X_val)
                mse = float(mean_squared_error(self.y_val, preds))
            except Exception as exc:
                if self.verbose:
                    print(f"    skip {delta}: {exc}")
                continue

            if mse < best["mse"]:
                best = {"mse": mse, "config": cfg, "delta": delta}

        return best

    def _fit_xgb_with_early_stopping(
        self,
        cfg: Dict,
        early_stopping_rounds: int = 20,
    ):
        """
        Fit XGB with early stopping on val set. Returns (best_iteration, val_mse).

        Handles both modern API (early_stopping_rounds in constructor) and
        legacy API (in fit kwargs).
        """
        # Try the modern constructor-arg API first
        try:
            est = XGBRegressor(
                **{**cfg, "early_stopping_rounds": early_stopping_rounds}
            )
            est.fit(self.X_train, self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    verbose=False)
        except TypeError:
            # Older xgboost versions accept it in fit() kwargs
            est = XGBRegressor(**cfg)
            est.fit(self.X_train, self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False)

        # best_iteration is 0-indexed; n_estimators should be best_iteration + 1
        best_iter = getattr(est, "best_iteration", None)
        n_est = (best_iter + 1) if best_iter is not None else cfg["n_estimators"]
        n_est = max(1, int(n_est))

        # Refit a clean estimator at the chosen n_estimators (no early stopping
        # arg), so val MSE is computed on the canonical config that the
        # bootstrap will use.
        clean_cfg = copy.deepcopy(cfg)
        clean_cfg["n_estimators"] = n_est
        for k in ("early_stopping_rounds",):
            clean_cfg.pop(k, None)
        eval_est = XGBRegressor(**clean_cfg)
        eval_est.fit(self.X_train, self.y_train)
        preds = eval_est.predict(self.X_val)
        mse = float(mean_squared_error(self.y_val, preds))
        return n_est, mse


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _grid_size(grid: Dict[str, List]) -> int:
    n = 1
    for v in grid.values():
        n *= len(v)
    return n
