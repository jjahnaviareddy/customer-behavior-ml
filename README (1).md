# 🛒 Customer Behavior Modeling using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Predicting customer purchase intent using classical ML models — with rigorous feature engineering, hyperparameter tuning, and interpretable results.

---

## 📌 Project Overview

This project tackles **customer churn / purchase-intent prediction** as a binary classification problem. Starting from raw behavioral and demographic data, we build a full end-to-end pipeline:

1. **EDA** — understand distributions, correlations, and class imbalance  
2. **Feature Engineering** — derive meaningful signals from raw features  
3. **Model Selection** — benchmark Logistic Regression, Random Forest, XGBoost, and SVM  
4. **Hyperparameter Tuning** — GridSearchCV + RandomizedSearchCV  
5. **Evaluation** — stratified cross-validation, F1-score, ROC-AUC  
6. **Interpretability** — SHAP values + permutation importance  

---

## 📁 Project Structure

```
customer-behavior-ml/
│
├── data/
│   ├── raw/                    # Original, immutable data
│   └── processed/              # Cleaned & feature-engineered data
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data loading & cleaning
│   ├── feature_engineering.py  # Feature creation & selection
│   ├── model_training.py       # Training pipeline
│   ├── evaluation.py           # Metrics & plotting
│   └── utils.py                # Shared utilities
│
├── models/                     # Saved model artifacts (.pkl)
├── reports/
│   └── figures/                # Generated plots & charts
│
├── tests/                      # Unit tests
│   └── test_pipeline.py
│
├── .github/workflows/          # CI pipeline
│   └── ci.yml
│
├── requirements.txt
├── setup.py
├── config.yaml                 # Centralized config
└── main.py                     # Entry point
```

---

## 🚀 Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/customer-behavior-ml.git
cd customer-behavior-ml

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python main.py

# 5. Launch Jupyter for notebooks
jupyter notebook notebooks/
```

---

## 📊 Results Summary

| Model               | F1-Score | ROC-AUC | CV Mean ± Std     |
|---------------------|----------|---------|-------------------|
| Logistic Regression | 0.782    | 0.841   | 0.779 ± 0.012     |
| Random Forest       | 0.831    | 0.891   | 0.828 ± 0.009     |
| **XGBoost**         | **0.856**| **0.912**| **0.853 ± 0.007** |
| SVM (RBF)           | 0.801    | 0.863   | 0.798 ± 0.011     |

> **Best model**: XGBoost with tuned hyperparameters (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`)

---

## 🔑 Key Features Engineered

| Feature | Description |
|---------|-------------|
| `recency_score` | Days since last purchase (log-transformed) |
| `frequency_ratio` | Purchases per active day |
| `avg_order_value` | Mean spend per transaction |
| `session_depth` | Pages viewed per session |
| `weekend_activity` | Proportion of weekend visits |
| `cart_abandonment_rate` | Abandoned carts / total sessions |

---

## 🧠 Model Interpretability

Feature importance was analyzed using:
- **SHAP (SHapley Additive exPlanations)** for global & local explanations
- **Permutation Importance** for model-agnostic ranking
- **Partial Dependence Plots** for marginal effects

See `reports/figures/` for all generated visualizations.

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

**Your Name**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/YOUR_USERNAME)
