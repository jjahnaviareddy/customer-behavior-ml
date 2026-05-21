"""
tests/test_pipeline.py
-----------------------
Unit tests for preprocessing, feature engineering, and model components.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import generate_synthetic_data, clean_data, split_data
from src.feature_engineering import CustomerFeatureEngineer, CategoricalEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_df():
    return generate_synthetic_data(n_samples=500, random_state=0)


@pytest.fixture
def clean_df(raw_df):
    return clean_data(raw_df)


# ---------------------------------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------------------------------

class TestDataGeneration:
    def test_shape(self, raw_df):
        assert raw_df.shape == (500, 15)

    def test_target_binary(self, raw_df):
        assert set(raw_df["purchased"].unique()).issubset({0, 1})

    def test_no_all_null_columns(self, raw_df):
        assert raw_df.isnull().all().sum() == 0

    def test_positive_rate_reasonable(self, raw_df):
        rate = raw_df["purchased"].mean()
        assert 0.05 <= rate <= 0.60, f"Positive rate out of range: {rate:.2%}"


class TestCleaning:
    def test_no_duplicates(self, clean_df):
        assert clean_df.duplicated().sum() == 0

    def test_no_nulls(self, clean_df):
        assert clean_df.isnull().sum().sum() == 0

    def test_shape_preserved(self, raw_df, clean_df):
        # Cleaning may drop rows, but columns stay the same
        assert clean_df.shape[1] == raw_df.shape[1]


class TestSplit:
    def test_proportions(self, clean_df):
        X_tr, X_te, y_tr, y_te = split_data(clean_df, test_size=0.2)
        total = len(X_tr) + len(X_te)
        assert abs(len(X_te) / total - 0.2) < 0.02

    def test_stratification(self, clean_df):
        _, _, y_tr, y_te = split_data(clean_df, test_size=0.2)
        assert abs(y_tr.mean() - y_te.mean()) < 0.05


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineer:
    def test_new_features_exist(self, clean_df):
        eng = CustomerFeatureEngineer()
        X = clean_df.drop(columns=["purchased"])
        X_out = eng.fit_transform(X)
        expected = [
            "recency_score", "frequency_ratio", "avg_order_value",
            "session_depth", "weekend_activity", "cart_abandonment_rate",
            "clv_proxy", "engagement_score",
        ]
        for feat in expected:
            assert feat in X_out.columns, f"Missing feature: {feat}"

    def test_no_inf_values(self, clean_df):
        eng = CustomerFeatureEngineer()
        X = clean_df.drop(columns=["purchased"])
        X_out = eng.fit_transform(X)
        num_cols = X_out.select_dtypes(include="number")
        assert not np.isinf(num_cols.values).any()

    def test_cart_abandonment_in_range(self, clean_df):
        eng = CustomerFeatureEngineer()
        X = clean_df.drop(columns=["purchased"])
        X_out = eng.fit_transform(X)
        assert (X_out["cart_abandonment_rate"] >= 0).all()
        assert (X_out["cart_abandonment_rate"] <= 1).all()


class TestCategoricalEncoder:
    def test_no_object_columns_after_encode(self, clean_df):
        eng = CustomerFeatureEngineer()
        enc = CategoricalEncoder()
        X = clean_df.drop(columns=["purchased"])
        X_eng = eng.fit_transform(X)
        X_enc = enc.fit_transform(X_eng)
        assert X_enc.select_dtypes(include="object").shape[1] == 0

    def test_train_test_column_alignment(self, clean_df):
        eng = CustomerFeatureEngineer()
        enc = CategoricalEncoder()
        X = clean_df.drop(columns=["purchased"])
        X_eng = eng.fit_transform(X)
        enc.fit(X_eng)
        X_train_enc = enc.transform(X_eng.iloc[:400])
        X_test_enc  = enc.transform(X_eng.iloc[400:])
        assert list(X_train_enc.columns) == list(X_test_enc.columns)
