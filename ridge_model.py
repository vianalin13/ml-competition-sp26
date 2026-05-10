"""
Ridge regression model for the CSI500 ensemble.

This model is intentionally simple and regularized. It is used as a stabilizing
linear component in the ensemble, not as the main alpha engine.
"""

from __future__ import annotations

import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


RIDGE_CONFIG = {
    "ridge": {
        "top_k": 75,
        "model_type": "ridge",
        "params": {
            "alpha": 10.0,
        },
    },
}


def train_ridge_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    alpha: float = 10.0,
):
    """
    Train a Ridge regression model.

    StandardScaler is important because Ridge penalizes coefficient size,
    so features need to be on comparable scales.
    """
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )

    model.fit(
        train_df[feature_columns],
        train_df[target_column],
    )

    return model