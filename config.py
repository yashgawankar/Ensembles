"""
config.py — Global constants, baseline hyperparameters, and experiment definitions.
All values here serve as the single source of truth for the pipeline.
"""

import numpy as np

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
GLOBAL_SEED = 42

# ─────────────────────────────────────────────
# Dataset defaults (synthetic)
# ─────────────────────────────────────────────
DATASET_CONFIG = {
    "n_total": 10_000,
    "n_test": 2_000,
    "n_train_pool": 8_000,
    "n_features": 15,
    "n_informative": 8,
    "n_noise_features": 7,
    "default_noise": 15.0,
}

# ─────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────
BOOTSTRAP_CONFIG = {
    "n_bootstrap": 50,
    "n_prob_hybrid_runs": 100,  # Extra averaging for probabilistic hybrid variance stability
}

# ─────────────────────────────────────────────
# Baseline hyperparameters
# ─────────────────────────────────────────────
BASELINE_PARAMS = {
    "rf": {
        "n_estimators": 100,
        "max_depth": 10,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "bootstrap": True,
        "random_state": GLOBAL_SEED,
        "n_jobs": -1,
    },
    "xgb": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "objective": "reg:squarederror",
        "random_state": GLOBAL_SEED,
        "n_jobs": -1,
    },
    "bagged_xgb": {
        "outer": {
            "n_estimators": 10,
            "max_samples": 1.0,
            "bootstrap": True,
            "random_state": GLOBAL_SEED,
            "n_jobs": -1,
        },
        "inner": {
            "n_estimators": 50,
            "max_depth": 5,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "random_state": GLOBAL_SEED,
        },
    },
}

# ─────────────────────────────────────────────
# Experiment parameter sweeps
# ─────────────────────────────────────────────
EXPERIMENT_VARIATIONS = {
    "ensemble_size": {
        "label": "Ensemble Size",
        "rf_param": "n_estimators",
        "xgb_param": "n_estimators",
        "bagged_xgb_param": "outer_n_estimators",
        "values": [5, 10, 25, 50, 100, 200],
        "n_train": 4000,
        "noise": 15.0,
    },
    "depth": {
        "label": "Base Learner Depth",
        "rf_param": "max_depth",
        "xgb_param": "max_depth",
        "bagged_xgb_param": "inner_max_depth",
        "values": [2, 4, 6, 8, 10, 15, 20],  # None (unlimited) added only for RF
        "values_rf_extra": [None],             # RF-only unlimited depth
        "n_train": 4000,
        "noise": 15.0,
    },
    "noise": {
        "label": "Noise Level (σ)",
        "values": [5, 15, 30, 50, 75],
        "n_train": 4000,
    },
    "train_size": {
        "label": "Training Set Size",
        "values": [500, 1000, 2000, 4000, 8000],
        "noise": 15.0,
    },
}

# ─────────────────────────────────────────────
# Hybrid sweep
# ─────────────────────────────────────────────
LAMBDA_VALUES = [round(v, 1) for v in np.linspace(0.0, 1.0, 11)]

# ─────────────────────────────────────────────
# Theory validation conditions
# ─────────────────────────────────────────────
THEORY_CONDITIONS = {
    "baseline":    {"noise": 15.0, "n_train": 4000},
    "low_noise":   {"noise": 5.0,  "n_train": 4000},
    "high_noise":  {"noise": 50.0, "n_train": 4000},
    "small_data":  {"noise": 15.0, "n_train": 500},
    "large_data":  {"noise": 15.0, "n_train": 8000},
}

# ─────────────────────────────────────────────
# Colour palette for plots
# ─────────────────────────────────────────────
COLORS = {
    "rf":         "#2563EB",   # blue
    "xgb":        "#DC2626",   # red
    "bagged_xgb": "#7C3AED",  # purple
    "convex":     "#059669",   # emerald
    "probabilistic": "#D97706", # amber
    "iso_contour": "#94A3B8",  # slate (dashed)
}

MODEL_LABELS = {
    "rf":         "Random Forest",
    "xgb":        "XGBoost",
    "bagged_xgb": "Bagged-XGB",
}
