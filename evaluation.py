"""
src/evaluation.py
-----------------
Metrics computation, comparison table, and all plot generation.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

logger = logging.getLogger(__name__)
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "logistic_regression": "#4C72B0",
    "random_forest":        "#55A868",
    "xgboost":              "#C44E52",
    "svm":                  "#DD8452",
}


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Compute all evaluation metrics for a single model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "avg_prec":  average_precision_score(y_test, y_prob),
        "precision": float(classification_report(y_test, y_pred, output_dict=True)["1"]["precision"]),
        "recall":    float(classification_report(y_test, y_pred, output_dict=True)["1"]["recall"]),
    }
    return metrics, y_prob


def compare_models(
    models: dict[str, Any],
    X_test,
    y_test,
) -> pd.DataFrame:
    """Evaluate all models and return a sorted comparison DataFrame."""
    rows = []
    for name, model in models.items():
        metrics, _ = evaluate_model(model, X_test, y_test)
        rows.append({"Model": name, **metrics})

    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    logger.info("\n" + df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_curves(models: dict, X_test, y_test, save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=PALETTE.get(name), linewidth=2)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "roc_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    return fig


def plot_precision_recall(models: dict, X_test, y_test, save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})",
                color=PALETTE.get(name), linewidth=2)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "precision_recall_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(model, X_test, y_test, model_name: str = "", save: bool = True):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No Purchase", "Purchase"],
        yticklabels=["No Purchase", "Purchase"],
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    fig.tight_layout()

    if save:
        safe_name = model_name.replace(" ", "_").lower()
        path = FIGURES_DIR / f"cm_{safe_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(model, feature_names: list[str], model_name: str = "", top_n: int = 15, save: bool = True):
    """Works for tree-based models with feature_importances_ attribute."""
    if not hasattr(model, "feature_importances_"):
        logger.warning(f"{model_name} has no feature_importances_. Skipping.")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in indices]
    vals  = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]
    bars = ax.barh(names[::-1], vals[::-1], color=colors)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if save:
        safe_name = model_name.replace(" ", "_").lower()
        path = FIGURES_DIR / f"feature_importance_{safe_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    return fig


def plot_metrics_comparison(comparison_df: pd.DataFrame, save: bool = True):
    metrics = ["f1", "roc_auc", "precision", "recall"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5), sharey=False)

    for ax, metric in zip(axes, metrics):
        colors = [PALETTE.get(m, "#888") for m in comparison_df["Model"]]
        bars = ax.bar(comparison_df["Model"], comparison_df[metric], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(metric.upper().replace("_", " "), fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_xticklabels(comparison_df["Model"], rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Model Comparison — Key Metrics", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "model_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    return fig
