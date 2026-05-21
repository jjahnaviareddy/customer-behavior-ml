"""
main.py
--------
End-to-end pipeline entry point.

Usage
-----
    python main.py                      # full run with tuning
    python main.py --no-tune            # skip hyperparameter search
    python main.py --config custom.yaml # custom config file
"""

import argparse
import logging

import pandas as pd

from src.utils import setup_logging, load_config, set_seed, ensure_dirs, print_section
from src.data_preprocessing import load_data, clean_data, split_data, get_scaler, apply_smote
from src.feature_engineering import CustomerFeatureEngineer, CategoricalEncoder
from src.model_training import train_all_models, save_model
from src.evaluation import (
    compare_models,
    plot_roc_curves,
    plot_precision_recall,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metrics_comparison,
)

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Customer Behavior ML Pipeline")
    parser.add_argument("--config",   default="config.yaml",      help="Path to config YAML")
    parser.add_argument("--no-tune",  action="store_true",         help="Skip hyperparameter tuning")
    parser.add_argument("--no-plots", action="store_true",         help="Skip plot generation")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    set_seed(cfg["data"]["random_state"])
    ensure_dirs(cfg["output"]["model_dir"], cfg["output"]["report_dir"])

    # ── 1. Load & Clean ──────────────────────────────────────────────────────
    print_section("1. Data Loading & Cleaning")
    df = load_data(cfg["data"]["raw_path"])
    df = clean_data(df)
    logger.info(f"Dataset shape: {df.shape} | Positive rate: {df[cfg['data']['target_column']].mean():.2%}")

    # ── 2. Feature Engineering ───────────────────────────────────────────────
    print_section("2. Feature Engineering")
    engineer = CustomerFeatureEngineer()
    df_feat  = engineer.fit_transform(df.drop(columns=[cfg["data"]["target_column"]]))
    df_feat[cfg["data"]["target_column"]] = df[cfg["data"]["target_column"]].values

    encoder  = CategoricalEncoder()
    features = df_feat.drop(columns=[cfg["data"]["target_column"]])
    features = encoder.fit_transform(features)
    logger.info(f"Engineered feature matrix: {features.shape}")

    # ── 3. Split & Balance ───────────────────────────────────────────────────
    print_section("3. Train/Test Split & Class Balancing")
    target = cfg["data"]["target_column"]
    full_df = features.copy()
    full_df[target] = df_feat[target].values

    X_train, X_test, y_train, y_test = split_data(
        full_df,
        target=target,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )

    if cfg["preprocessing"]["handle_imbalance"]:
        X_train_num = X_train.select_dtypes(include="number")
        X_train_num, y_train = apply_smote(X_train_num, y_train)
        X_train = pd.DataFrame(X_train_num, columns=X_train.select_dtypes(include="number").columns)

    # Scale numeric features
    scaler       = get_scaler(cfg["preprocessing"]["scaling"])
    X_train_sc   = scaler.fit_transform(X_train)
    X_test_sc    = scaler.transform(X_test.select_dtypes(include="number"))
    feature_names = X_train.columns.tolist()

    # ── 4. Train & Tune ──────────────────────────────────────────────────────
    print_section("4. Model Training & Hyperparameter Tuning")
    models = train_all_models(X_train_sc, y_train, tune=not args.no_tune)

    # ── 5. Evaluate ──────────────────────────────────────────────────────────
    print_section("5. Evaluation on Hold-out Test Set")
    comparison = compare_models(models, X_test_sc, y_test)
    print("\n" + comparison.to_string(index=False))

    # Save comparison
    comparison.to_csv("reports/model_comparison.csv", index=False)

    # ── 6. Save Best Model ───────────────────────────────────────────────────
    best_name  = comparison.iloc[0]["Model"]
    best_model = models[best_name]
    save_model(best_model, cfg["output"]["best_model_path"])
    logger.info(f"Best model: {best_name}")

    # Save all models
    for name, model in models.items():
        save_model(model, f"{cfg['output']['model_dir']}{name}.pkl")

    # ── 7. Plots ─────────────────────────────────────────────────────────────
    if not args.no_plots:
        print_section("6. Generating Plots")
        plot_roc_curves(models, X_test_sc, y_test)
        plot_precision_recall(models, X_test_sc, y_test)
        plot_metrics_comparison(comparison)
        plot_confusion_matrix(best_model, X_test_sc, y_test, model_name=best_name)
        for name, model in models.items():
            plot_feature_importance(model, feature_names, model_name=name)

    print_section("✅ Pipeline Complete")
    logger.info("All outputs saved to reports/ and models/")


if __name__ == "__main__":
    main()
