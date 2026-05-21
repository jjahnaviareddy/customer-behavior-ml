"""
src/feature_engineering.py
---------------------------
Creates domain-driven features from raw behavioral data and encodes categoricals.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class CustomerFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer that engineers customer behavior features.

    New features
    ------------
    recency_score         : log-inverse of recency (higher = more recent)
    frequency_ratio       : purchases per session
    avg_order_value       : monetary value / frequency
    session_depth         : pages_per_session * session_count
    weekend_activity      : weekend_visits / session_count
    cart_abandonment_rate : cart_abandoned / (cart_events + 1)
    clv_proxy             : frequency × avg_order_value / (recency_days + 1)
    engagement_score      : composite of time, pages, sessions
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Guard: avoid division by zero
        eps = 1e-6

        df["recency_score"] = np.log1p(1 / (df["recency_days"] + eps))
        df["frequency_ratio"] = df["frequency"] / (df["session_count"] + eps)
        df["avg_order_value"] = df["monetary_value"] / (df["frequency"] + eps)
        df["session_depth"] = df["pages_per_session"] * df["session_count"]
        df["weekend_activity"] = df["weekend_visits"] / (df["session_count"] + eps)
        df["cart_abandonment_rate"] = df["cart_abandoned"] / (df["cart_events"] + 1)
        df["clv_proxy"] = (
            df["frequency"] * df["avg_order_value"] / (df["recency_days"] + 1)
        )
        df["engagement_score"] = (
            np.log1p(df["avg_time_on_site"])
            * np.log1p(df["pages_per_session"])
            * np.log1p(df["session_count"])
        )

        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode categorical columns; drop first to avoid multicollinearity."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns
        self._dummies_columns: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns or X.select_dtypes(include="object").columns.tolist()
        dummy_df = pd.get_dummies(X[cols], drop_first=True)
        self._dummies_columns = dummy_df.columns.tolist()
        self._cat_cols = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        dummy_df = pd.get_dummies(df[self._cat_cols], drop_first=True)
        # Align columns to training schema
        dummy_df = dummy_df.reindex(columns=self._dummies_columns, fill_value=0)
        df = df.drop(columns=self._cat_cols)
        df = pd.concat([df, dummy_df], axis=1)
        return df


def select_features(X: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """Return only the specified columns (safe subset)."""
    available = [f for f in feature_list if f in X.columns]
    missing = set(feature_list) - set(available)
    if missing:
        logger.warning(f"Missing requested features: {missing}")
    return X[available]
