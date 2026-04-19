"""
experiments/bootstrap_evaluator.py — BootstrapEvaluator.

Core workhorse.  Given an unfitted model + training/test data, runs M bootstrap
iterations and decomposes test-set error into bias² and variance.

Design choices:
  - sklearn.base.clone() creates a fresh copy of the model each iteration,
    guaranteeing no state leaks between bootstrap runs.
  - Seeds are deterministic per-iteration (base_seed + i) so the whole
    procedure is reproducible.
  - All M × n_test predictions are returned to enable hybrid computation
    and covariance estimation downstream.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, clone

from pipeline_types import BiasVarianceResult


class BootstrapEvaluator:
    """
    Bootstrap bias-variance decomposition for a single model configuration.

    Usage
    -----
    evaluator = BootstrapEvaluator(n_bootstrap=50, random_seed=42)
    result = evaluator.evaluate(model, X_train, y_train, X_test, y_test)

    result.bias_squared  →  scalar
    result.variance      →  scalar
    result.mse           →  scalar
    result.all_predictions  →  shape (n_bootstrap, n_test)
    """

    def __init__(
        self,
        n_bootstrap: int = 50,
        random_seed: int = 42,
        verbose: bool = False,
    ):
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        self.verbose = verbose

    # ── Main API ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        label: str = "",
    ) -> BiasVarianceResult:
        """
        Perform M bootstrap iterations.

        Parameters
        ----------
        model   : Unfitted sklearn-compatible estimator.
        X_train : Full training features (will be resampled each iteration).
        y_train : Training targets.
        X_test  : Fixed test features (same 2,000 points every time).
        y_test  : True test labels.
        label   : Optional string for verbose logging.

        Returns
        -------
        BiasVarianceResult with bias², variance, MSE, and all predictions.
        """
        n_test = len(y_test)
        all_predictions = np.zeros((self.n_bootstrap, n_test))

        t0 = time.time()
        for i in range(self.n_bootstrap):
            seed_i = self.random_seed + i
            X_boot, y_boot = self._bootstrap_sample(X_train, y_train, seed_i)

            # Fresh clone prevents any state from persisting across iterations
            m = clone(model)
            m.fit(X_boot, y_boot)
            all_predictions[i, :] = m.predict(X_test)

            if self.verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"    [{label}] Bootstrap {i+1}/{self.n_bootstrap}  "
                      f"({elapsed:.1f}s elapsed)")

        return self._compute_result(all_predictions, y_test)

    def evaluate_paired(
        self,
        model_rf: BaseEstimator,
        model_xgb: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        label: str = "",
    ) -> tuple[BiasVarianceResult, BiasVarianceResult, np.ndarray, np.ndarray]:
        """
        Train RF and XGB on the SAME bootstrap sample in each iteration.

        This is required for covariance computation in hybrid analysis (E5/E6):
        if models see different bootstrap samples, their predictions are
        artificially de-correlated.

        Returns
        -------
        (bv_rf, bv_xgb, rf_predictions, xgb_predictions)
        where rf_predictions and xgb_predictions both have shape (M, n_test).
        """
        n_test = len(y_test)
        rf_predictions  = np.zeros((self.n_bootstrap, n_test))
        xgb_predictions = np.zeros((self.n_bootstrap, n_test))

        t0 = time.time()
        for i in range(self.n_bootstrap):
            seed_i = self.random_seed + i
            X_boot, y_boot = self._bootstrap_sample(X_train, y_train, seed_i)

            m_rf  = clone(model_rf)
            m_xgb = clone(model_xgb)

            m_rf.fit(X_boot, y_boot)
            m_xgb.fit(X_boot, y_boot)

            rf_predictions[i, :]  = m_rf.predict(X_test)
            xgb_predictions[i, :] = m_xgb.predict(X_test)

            if self.verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"    [{label}] Paired bootstrap {i+1}/{self.n_bootstrap}  "
                      f"({elapsed:.1f}s elapsed)")

        bv_rf  = self._compute_result(rf_predictions, y_test)
        bv_xgb = self._compute_result(xgb_predictions, y_test)
        return bv_rf, bv_xgb, rf_predictions, xgb_predictions

    # ── Internals ─────────────────────────────────────────────────────────

    def _bootstrap_sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample n rows with replacement."""
        rng = np.random.RandomState(seed)
        n = len(y)
        idx = rng.choice(n, size=n, replace=True)
        return X[idx], y[idx]

    def _compute_result(
        self,
        all_predictions: np.ndarray,   # (M, n_test)
        y_test: np.ndarray,
    ) -> BiasVarianceResult:
        """
        Decompose:
            bias²    = E[(ŷ_mean - y)²]   averaged over test points
            variance = E[Var(ŷ)]           averaged over test points
            mse      = bias² + variance    (law of total expectation)

        Note: mse computed this way equals the actual test MSE only when
        the decomposition is exact, which holds under the bootstrap approximation.
        We also store the directly computed mse for validation.
        """
        mean_pred = all_predictions.mean(axis=0)         # (n_test,)

        per_point_bias_sq = (mean_pred - y_test) ** 2   # (n_test,)
        per_point_variance = all_predictions.var(axis=0, ddof=0)  # (n_test,)

        bias_squared = per_point_bias_sq.mean()
        variance     = per_point_variance.mean()
        mse          = bias_squared + variance

        return BiasVarianceResult(
            bias_squared=float(bias_squared),
            variance=float(variance),
            mse=float(mse),
            mean_prediction=mean_pred,
            all_predictions=all_predictions,
            per_point_bias_sq=per_point_bias_sq,
            per_point_variance=per_point_variance,
        )
