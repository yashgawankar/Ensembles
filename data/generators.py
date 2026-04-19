"""
data/generators.py — Abstract base + concrete DataGenerator implementations.

Supports:
  - SyntheticDataGenerator  (parametric regression with nonlinear terms)
  - CaliforniaHousingDataGenerator
  - CustomCSVDataGenerator

All generators return raw (X, y) and metadata.  Splitting / scaling happens
in DataPreprocessor, not here, so generators stay pure.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, make_regression


# ─────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────

class DataGenerator(ABC):
    """Contract that every data source must satisfy."""

    @abstractmethod
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) as raw numpy arrays (unscaled, unsplit)."""

    @abstractmethod
    def get_info(self) -> Dict:
        """Return metadata describing the dataset."""


# ─────────────────────────────────────────────
# Synthetic dataset
# ─────────────────────────────────────────────

class SyntheticDataGenerator(DataGenerator):
    """
    Generates:
        y = linear_component + nonlinear_component + ε

    where:
        linear_component    = weighted combination of 8 informative features
        nonlinear_component = 3·sin(2·x₁) + 2·x₂² + 1.5·x₃·x₄
        ε                   ~ N(0, noise_level²)

    Keeps the deterministic components fixed across different noise_level
    instantiations so that experiments 3 (noise sweep) are directly comparable.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        n_features: int = 15,
        n_informative: int = 8,
        noise_level: float = 15.0,
        random_seed: int = 42,
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.noise_level = noise_level
        self.random_seed = random_seed

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.random_seed)

        # ── Linear component via make_regression (same seed every time) ───
        X, y_linear, _ = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            noise=0.0,            # We add noise ourselves
            coef=True,
            random_state=self.random_seed,
        )

        # ── Nonlinear component (uses first 4 informative features) ───────
        y_nonlinear = (
            3.0 * np.sin(2.0 * X[:, 0])
            + 2.0 * X[:, 1] ** 2
            + 1.5 * X[:, 2] * X[:, 3]
        )

        # ── Noise (drawn independently; seed is separate so noise varies) ─
        noise_rng = np.random.RandomState(self.random_seed + 999)
        epsilon = noise_rng.normal(0, self.noise_level, size=self.n_samples)

        y = y_linear + y_nonlinear + epsilon
        return X, y

    def generate_with_noise(self, noise_level: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Same X and deterministic y as generate(), but with a different noise level.
        Used by DatasetVariantManager for noise sweep (E3).
        """
        rng = np.random.RandomState(self.random_seed)

        X, y_linear, _ = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            noise=0.0,
            coef=True,
            random_state=self.random_seed,
        )
        y_nonlinear = (
            3.0 * np.sin(2.0 * X[:, 0])
            + 2.0 * X[:, 1] ** 2
            + 1.5 * X[:, 2] * X[:, 3]
        )

        # New noise drawn with noise_level-specific seed for reproducibility
        noise_seed = self.random_seed + int(noise_level * 100)
        noise_rng = np.random.RandomState(noise_seed)
        epsilon = noise_rng.normal(0, noise_level, size=self.n_samples)

        y = y_linear + y_nonlinear + epsilon
        return X, y

    def get_info(self) -> Dict:
        return {
            "type": "synthetic",
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_informative": self.n_informative,
            "n_noise_features": self.n_features - self.n_informative,
            "noise_level": self.noise_level,
            "target": "linear + 3·sin(2x₁) + 2x₂² + 1.5x₃x₄ + ε",
        }


# ─────────────────────────────────────────────
# California Housing (real-world)
# ─────────────────────────────────────────────

class CaliforniaHousingDataGenerator(DataGenerator):
    """
    Loads sklearn's California Housing dataset.
    n_samples ≈ 20,640, n_features = 8.

    No noise variants – noise is inherent in real data.
    """

    def __init__(self):
        self._data = None

    def _load(self):
        if self._data is None:
            self._data = fetch_california_housing(as_frame=False)

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        self._load()
        return self._data.data.copy(), self._data.target.copy()

    def get_info(self) -> Dict:
        self._load()
        return {
            "type": "california_housing",
            "n_samples": self._data.data.shape[0],
            "n_features": self._data.data.shape[1],
            "feature_names": list(self._data.feature_names),
            "target": "Median house value (100k USD)",
            "noise_variants_supported": False,
        }


# ─────────────────────────────────────────────
# Generic CSV loader
# ─────────────────────────────────────────────

class CustomCSVDataGenerator(DataGenerator):
    """
    Loads any CSV file where the target column is specified by name.

    Parameters
    ----------
    filepath    : path to CSV file
    target_col  : name of the target column
    drop_cols   : list of column names to drop (IDs, dates, etc.)
    """

    def __init__(
        self,
        filepath: str,
        target_col: str,
        drop_cols: list[str] | None = None,
    ):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV not found: {filepath}")
        self.filepath = filepath
        self.target_col = target_col
        self.drop_cols = drop_cols or []
        self._df = None

    def _load(self):
        if self._df is None:
            self._df = pd.read_csv(self.filepath)

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        self._load()
        df = self._df.drop(columns=self.drop_cols, errors="ignore")
        y = df[self.target_col].values.astype(float)
        X = df.drop(columns=[self.target_col]).select_dtypes(include=[np.number]).values.astype(float)
        return X, y

    def get_info(self) -> Dict:
        self._load()
        return {
            "type": "custom_csv",
            "filepath": self.filepath,
            "n_samples": len(self._df),
            "target_col": self.target_col,
            "drop_cols": self.drop_cols,
        }
