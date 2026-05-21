"""
src/data_preprocessing.py
--------------------------
Handles data loading, cleaning, and train/test splitting.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Generation (synthetic dataset for demo purposes)
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic customer behavior dataset.

    Returns
    -------
    pd.DataFrame
        Raw customer behavioral and demographic features with binary target.
    """
    rng = np.random.default_rng(random_state)

    n = n_samples
    data = {
        # Behavioral features
        "recency_days":       rng.integers(1, 365, n),
        "frequency":          rng.integers(1, 50, n),
        "monetary_value":     rng.exponential(scale=150, size=n).round(2),
        "session_count":      rng.integers(1, 100, n),
        "pages_per_session":  rng.uniform(1.0, 20.0, n).round(2),
        "avg_time_on_site":   rng.uniform(30, 1800, n).round(0),   # seconds
        "cart_events":        rng.integers(0, 20, n),
        "cart_abandoned":     rng.integers(0, 10, n),
        "weekend_visits":     rng.integers(0, 30, n),

        # Demographic / acquisition
        "customer_age":       rng.integers(18, 70, n),
        "account_age_days":   rng.integers(1, 2000, n),
        "device_type":        rng.choice(["mobile", "desktop", "tablet"], n, p=[0.55, 0.35, 0.10]),
        "traffic_source":     rng.choice(["organic", "paid", "social", "email", "direct"], n,
                                          p=[0.30, 0.25, 0.20, 0.15, 0.10]),
        "customer_segment":   rng.choice(["new", "returning", "loyal", "at_risk"], n,
                                          p=[0.35, 0.30, 0.20, 0.15]),
    }

    df = pd.DataFrame(data)

    # Ensure cart_abandoned <= cart_events
    df["cart_abandoned"] = np.minimum(df["cart_abandoned"], df["cart_events"])

    # Synthetic target with realistic signal
    log_odds = (
        -1.5
        - 0.005 * df["recency_days"]
        + 0.04  * df["frequency"]
        + 0.002 * df["monetary_value"]
        + 0.02  * df["session_count"]
        + 0.05  * df["pages_per_session"]
        - 0.001 * df["avg_time_on_site"]
        - 0.3   * (df["cart_abandoned"] / (df["cart_events"] + 1))
        + 0.5   * (df["customer_segment"] == "loyal").astype(int)
        + 0.3   * (df["traffic_source"] == "email").astype(int)
        - 0.2   * (df["device_type"] == "mobile").astype(int)
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    df["purchased"] = (rng.uniform(size=n) < prob).astype(int)

    logger.info(f"Generated {n} samples | Purchase rate: {df['purchased'].mean():.2%}")
    return df


# ---------------------------------------------------------------------------
# Loading & Cleaning
# ---------------------------------------------------------------------------

def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load CSV data, with fallback to synthetic generation."""
    path = Path(filepath)
    if path.exists():
        df = pd.read_csv(path)
        logger.info(f"Loaded data from {path} — shape: {df.shape}")
    else:
        logger.warning(f"{path} not found. Generating synthetic dataset.")
        df = generate_synthetic_data()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Synthetic data saved to {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
      - Drop duplicates
      - Clip extreme outliers (winsorize at 1st/99th percentile)
      - Fill missing values
    """
    original_shape = df.shape
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = "purchased"
    numeric_features = [c for c in numeric_cols if c != target]

    for col in numeric_features:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)

    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    logger.info(f"Cleaned: {original_shape} → {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Splitting & Scaling
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    target: str = "purchased",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified train/test split."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Train positive rate: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


def get_scaler(strategy: str = "standard"):
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    return scalers.get(strategy, StandardScaler())


def apply_smote(X_train, y_train, random_state: int = 42):
    """Balance classes with SMOTE."""
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {X_res.shape} | Positive rate: {y_res.mean():.2%}")
    return X_res, y_res
