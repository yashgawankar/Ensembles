"""
data/preprocessor.py — DataPreprocessor and DatasetVariantManager.

Responsibilities:
  - Train / test split (fixed seed → same 2,000 test points always)
  - StandardScaler fitting on train only, applied to both
  - Subsampling for training-size variants (E4)
  - Noise variants for synthetic datasets (E3)
  - On-disk caching of processed splits
"""

from __future__ import annotations
import _path_setup

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.generators import DataGenerator, SyntheticDataGenerator


# ─────────────────────────────────────────────
# DataPreprocessor
# ─────────────────────────────────────────────

class DataPreprocessor:
    """
    Takes a DataGenerator and produces consistent, reproducible train/test splits.

    All scaling is fit ONLY on the training portion, then applied to both splits.
    The test set is always the same 2,000 points (fixed random_state).
    """

    def __init__(
        self,
        data_generator: DataGenerator,
        test_size: int = 2_000,
        val_size: int = 0,
        random_seed: int = 42,
        cache_dir: Optional[str] = None,
    ):
        self.data_generator = data_generator
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed
        self.cache_dir = cache_dir

        # Lazy-loaded; populated on first call to get_train_test_split
        self._X_train_raw: Optional[np.ndarray] = None
        self._X_test_raw:  Optional[np.ndarray] = None
        self._X_val_raw:   Optional[np.ndarray] = None
        self._y_train:     Optional[np.ndarray] = None
        self._y_test:      Optional[np.ndarray] = None
        self._y_val:       Optional[np.ndarray] = None
        self._scaler:      Optional[StandardScaler] = None

        self._initialised = False

    # ── Public API ────────────────────────────────────────────────────────

    def initialise(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        """
        Generate / load data and perform the primary train/test split + scaling.
        Can optionally accept pre-generated (X, y) to avoid double-generation.
        """
        if self._initialised:
            return  # Idempotent

        # Check disk cache first
        if self.cache_dir and self._load_from_cache():
            self._initialised = True
            return

        # Generate from source
        if X is None or y is None:
            X, y = self.data_generator.generate()

        n_total = len(X)
        test_frac = self.test_size / n_total

        X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
            X, y,
            test_size=test_frac,
            random_state=self.random_seed,
        )

        # Optional validation split carved out of the training pool BEFORE scaling.
        # Validation set uses a separate fixed seed so test set is unaffected.
        if self.val_size and self.val_size > 0:
            n_pool = len(y_tr)
            val_frac = min(self.val_size / n_pool, 0.5)
            X_pool_raw, X_val_raw, y_pool, y_val = train_test_split(
                X_tr_raw, y_tr,
                test_size=val_frac,
                random_state=self.random_seed + 1,
            )
        else:
            X_pool_raw, y_pool = X_tr_raw, y_tr
            X_val_raw, y_val   = None, None

        # Fit scaler on TRAIN POOL only (not on val or test)
        scaler = StandardScaler()
        X_pool_scaled = scaler.fit_transform(X_pool_raw)
        X_te_scaled   = scaler.transform(X_te_raw)
        X_val_scaled  = scaler.transform(X_val_raw) if X_val_raw is not None else None

        self._X_train_raw = X_pool_scaled   # "raw" but already scaled – term kept for clarity
        self._X_test_raw  = X_te_scaled
        self._X_val_raw   = X_val_scaled
        self._y_train     = y_pool
        self._y_test      = y_te
        self._y_val       = y_val
        self._scaler      = scaler

        if self.cache_dir:
            self._save_to_cache()

        self._initialised = True

    def get_train_test_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_train, X_test, y_train, y_test) – full training pool."""
        self._ensure_init()
        return self._X_train_raw.copy(), self._X_test_raw.copy(), self._y_train.copy(), self._y_test.copy()

    def get_train_subset(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a reproducible subset of the training data.
        Same seed → same indices every time, ensuring models trained on
        identical subsets for fair comparison.
        """
        self._ensure_init()
        n_available = len(self._y_train)

        if n_samples >= n_available:
            return self._X_train_raw.copy(), self._y_train.copy()

        rng = np.random.RandomState(self.random_seed + n_samples)
        idx = rng.choice(n_available, size=n_samples, replace=False)
        idx.sort()  # Keep natural ordering
        return self._X_train_raw[idx].copy(), self._y_train[idx].copy()

    @property
    def X_test(self) -> np.ndarray:
        self._ensure_init()
        return self._X_test_raw

    @property
    def y_test(self) -> np.ndarray:
        self._ensure_init()
        return self._y_test

    @property
    def X_val(self) -> Optional[np.ndarray]:
        self._ensure_init()
        return self._X_val_raw

    @property
    def y_val(self) -> Optional[np.ndarray]:
        self._ensure_init()
        return self._y_val

    @property
    def has_val_split(self) -> bool:
        self._ensure_init()
        return self._X_val_raw is not None and len(self._X_val_raw) > 0

    @property
    def scaler(self) -> StandardScaler:
        self._ensure_init()
        return self._scaler

    # ── Cache helpers ─────────────────────────────────────────────────────

    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, "base_split.pkl")

    def _save_to_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        payload = {
            "X_train":  self._X_train_raw,
            "X_test":   self._X_test_raw,
            "X_val":    self._X_val_raw,
            "y_train":  self._y_train,
            "y_test":   self._y_test,
            "y_val":    self._y_val,
            "scaler":   self._scaler,
            "val_size": self.val_size,
        }
        with open(self._cache_path(), "wb") as f:
            pickle.dump(payload, f)

    def _load_from_cache(self) -> bool:
        path = self._cache_path()
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            payload = pickle.load(f)
        # Cache schema check: rebuild if the requested val_size differs from cache
        cached_val = payload.get("val_size", 0)
        if cached_val != self.val_size:
            return False
        self._X_train_raw = payload["X_train"]
        self._X_test_raw  = payload["X_test"]
        self._X_val_raw   = payload.get("X_val")
        self._y_train     = payload["y_train"]
        self._y_test      = payload["y_test"]
        self._y_val       = payload.get("y_val")
        self._scaler      = payload["scaler"]
        return True

    def _ensure_init(self):
        if not self._initialised:
            self.initialise()


# ─────────────────────────────────────────────
# DatasetVariantManager
# ─────────────────────────────────────────────

class DatasetVariantManager:
    """
    Manages dataset variants for E3 (noise sweep) and E4 (train-size sweep).

    Variants share the same X and test set; only the training y (noise) or
    the number of training samples differs.
    """

    def __init__(
        self,
        preprocessor: DataPreprocessor,
        data_generator: Optional[DataGenerator] = None,
    ):
        self.preprocessor = preprocessor
        self.data_generator = data_generator  # needed for noise variants (synthetic only)
        self._variants: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

        # Always store the "full" baseline
        self._variants["full"] = preprocessor.get_train_test_split()

    # ── Public API ────────────────────────────────────────────────────────

    def create_noise_variants(self, noise_levels: List[float]):
        """
        For synthetic generators: re-draw ε with each noise level while keeping
        the deterministic components of y (linear + nonlinear) identical.

        For real datasets: this is a no-op (noise is inherent).
        """
        if not isinstance(self.data_generator, SyntheticDataGenerator):
            print("⚠  Noise variants only supported for SyntheticDataGenerator. Skipping.")
            return

        X_test = self.preprocessor.X_test
        y_test = self.preprocessor.y_test

        for sigma in noise_levels:
            key = f"noise_{sigma}"
            if key in self._variants:
                continue

            # generate_with_noise gives same X, same deterministic y, different ε
            X_full, y_full = self.data_generator.generate_with_noise(sigma)

            # Re-split using same seed so test indices are identical
            gen = self.data_generator
            n_total = len(X_full)
            test_frac = self.preprocessor.test_size / n_total

            from sklearn.model_selection import train_test_split
            X_tr_raw, _, y_tr, _ = train_test_split(
                X_full, y_full,
                test_size=test_frac,
                random_state=self.preprocessor.random_seed,
            )

            # Scale using the SAME scaler fitted on the baseline training split
            X_tr_scaled = self.preprocessor.scaler.transform(X_tr_raw)

            self._variants[key] = (X_tr_scaled, X_test, y_tr, y_test)

    def create_size_variants(self, train_sizes: List[int]):
        """Subsample the training pool to each requested size."""
        X_test = self.preprocessor.X_test
        y_test = self.preprocessor.y_test

        for n in train_sizes:
            key = f"size_{n}"
            if key in self._variants:
                continue
            X_sub, y_sub = self.preprocessor.get_train_subset(n)
            self._variants[key] = (X_sub, X_test, y_sub, y_test)

    def get_variant(
        self, variant_key: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve a dataset variant by key.

        Variant keys:
          'full'           → full training pool (baseline)
          'noise_<sigma>'  → noise variant for given σ
          'size_<n>'       → training subset of size n
        """
        if variant_key not in self._variants:
            available = list(self._variants.keys())
            raise KeyError(f"Variant '{variant_key}' not found. Available: {available}")
        return self._variants[variant_key]

    def list_variants(self) -> List[str]:
        return list(self._variants.keys())

    def get_baseline(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._variants["full"]
