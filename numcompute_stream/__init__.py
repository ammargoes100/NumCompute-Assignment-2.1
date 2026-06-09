"""
NumCompute-Stream.

A lightweight NumPy-based toolkit for streaming machine learning workflows,
including preprocessing, metrics, statistics, pipelines, decision trees,
tree ensembles, stream training, benchmarking, and visualisation.
"""

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.stream import StreamTrainer

from numcompute_stream.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
)

from numcompute_stream.pipeline import Pipeline, FeatureUnion

from numcompute_stream.metrics import (
    accuracy,
    precision,
    recall,
    f1,
    confusion_matrix,
    mse,
    roc_curve,
    auc,
    StreamingAccuracy,
    StreamingConfusionMatrix,
    StreamingPrecision,
    StreamingRecall,
    StreamingF1,
    StreamingMSE,
    RollingAccuracy,
)

from numcompute_stream.stats import (
    mean,
    median,
    std,
    minimum,
    maximum,
    histogram,
    quantile,
    WelfordStats,
    StreamingStats,
)

__all__ = [
    "DecisionTreeClassifier",
    "EnsembleClassifier",
    "StreamTrainer",
    "StandardScaler",
    "MinMaxScaler",
    "OneHotEncoder",
    "SimpleImputer",
    "Pipeline",
    "FeatureUnion",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "mse",
    "roc_curve",
    "auc",
    "StreamingAccuracy",
    "StreamingConfusionMatrix",
    "StreamingPrecision",
    "StreamingRecall",
    "StreamingF1",
    "StreamingMSE",
    "RollingAccuracy",
    "mean",
    "median",
    "std",
    "minimum",
    "maximum",
    "histogram",
    "quantile",
    "WelfordStats",
    "StreamingStats",
]