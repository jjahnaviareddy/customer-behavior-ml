"""
src/model_training.py
----------------------
Defines all classifiers, hyperparameter grids, tuning logic, and cross-validation.
"""

import logging
import joblib
import numpy as np
import yaml
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

def get_models() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=500, random_state=42),
        "random_forest":       RandomForestClassifier(random_state=42, n_jobs=-1),
        "xgboost":             XGBClassifier(
                                    eval_metric="logloss",
                                    random_state=42,
                                    n_jobs=-1,
                                    verbosity=0,
                               ),
        "svm":                 SVC(probability=True, random_state=42),
    }


def get_param_grids() -> dict[str, dict]:
    return {
        "logistic_regression": {
            "C":       [0.01, 0.1, 1, 10],
            "solver":  ["lbfgs", "liblinear"],
        },
        "random_forest": {
            "n_estimators":    [100, 200],
            "max_depth":       [4, 6, None],
            "min_samples_leaf":[1, 2, 4],
        },
        "xgboost": {
            "n_estimators":     [100, 200, 300],
            "max_depth":        [3, 5, 6],
            "learning_rate":    [0.01, 0.05, 0.1],
            "subsample":        [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
        "svm": {
            "C":      [0.1, 1, 10],
            "kernel": ["rbf", "poly"],
            "gamma":  ["scale", "auto"],
        },
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def cross_validate_model(
    model,
    X,
    y,
    scoring: list[str] | None = None,
) -> dict[str, float]:
    """5-fold stratified CV returning mean ± std for each metric."""
    if scoring is None:
        scoring = ["f1", "roc_auc", "precision", "recall"]

    results = cross_validate(model, X, y, cv=CV, scoring=scoring, n_jobs=-1)
    summary = {}
    for metric in scoring:
        vals = results[f"test_{metric}"]
        summary[metric] = {"mean": vals.mean(), "std": vals.std()}
        logger.info(f"  {metric}: {vals.mean():.4f} ± {vals.std():.4f}")
    return summary


def tune_model(
    model,
    param_grid: dict,
    X,
    y,
    strategy: str = "random",
    n_iter: int = 30,
    scoring: str = "f1",
) -> tuple[Any, dict]:
    """
    Hyperparameter search.

    Parameters
    ----------
    strategy : 'grid' | 'random'
    """
    if strategy == "grid":
        searcher = GridSearchCV(
            model, param_grid, cv=CV, scoring=scoring,
            n_jobs=-1, verbose=0, refit=True,
        )
    else:
        searcher = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=CV,
            scoring=scoring, n_jobs=-1, verbose=0,
            random_state=42, refit=True,
        )

    searcher.fit(X, y)
    logger.info(f"Best params: {searcher.best_params_}")
    logger.info(f"Best CV {scoring}: {searcher.best_score_:.4f}")
    return searcher.best_estimator_, searcher.best_params_


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train_all_models(X_train, y_train, tune: bool = True) -> dict[str, Any]:
    """
    Train (and optionally tune) every registered model.

    Returns
    -------
    dict mapping model name → fitted estimator
    """
    models = get_models()
    grids  = get_param_grids()
    trained = {}

    for name, model in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {name}")
        logger.info(f"{'='*50}")

        if tune:
            strategy = "grid" if name == "logistic_regression" else "random"
            best_model, _ = tune_model(
                model, grids[name], X_train, y_train,
                strategy=strategy, scoring="f1",
            )
        else:
            best_model = model.fit(X_train, y_train)

        logger.info(f"Cross-validating {name}...")
        cross_validate_model(best_model, X_train, y_train)
        trained[name] = best_model

    return trained


def save_model(model, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str | Path):
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model
