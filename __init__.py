"""
bias_variance_pipeline — Bagging vs Boosting Bias-Variance Analysis Pipeline.

Package structure:
    config.py       — global constants and baseline hyperparameters
    types.py        — shared dataclasses
    data/           — DataGenerator, DataPreprocessor, DatasetVariantManager
    models/         — HyperparameterRegistry, ModelBuilder
    experiments/    — BootstrapEvaluator, ExperimentOrchestrator,
                      HybridComputationEngine, TheoryValidator
    analysis/       — ResultsStore, MetricsComputer, TableBuilder
    visualization/  — TrajectoryPlotter, HybridPlotter
    main.py         — top-level execution script
"""
