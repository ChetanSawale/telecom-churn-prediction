"""Build notebook.ipynb: markdown narrative + runnable code cells + embedded chart outputs."""
import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CHARTS = os.path.join(HERE, "charts")


def img_cell(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "source": [],
        "outputs": [{
            "output_type": "display_data", "metadata": {},
            "data": {"image/png": b64, "text/plain": ["<Figure>"]},
        }],
    }


def md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "source": source.splitlines(keepends=True), "outputs": []}


cells = [
    md("# Telecom Customer Churn Prediction\n"
       "Supervised ML on a real business problem: predict which customers will churn "
       "so the retention team can act *before* they leave.\n\n"
       "**Data:** IBM Telco Customer Churn sample dataset — 7,043 customers, "
       "21 columns, 26.5% churn rate."),

    md("## 1. Load & clean"),
    code("""import pandas as pd, numpy as np
df = pd.read_csv("telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"{len(df):,} customers | churn rate {df['Churn'].eq('Yes').mean():.1%} | "
      f"blank TotalCharges: {df['TotalCharges'].isna().sum()}")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = (df["Churn"] == "Yes").astype(int)"""),

    md("## 2. EDA — where is churn concentrated?\n"
       "Month-to-month contracts churn at 3× the rate of two-year contracts. "
       "Churn is front-loaded: it collapses as tenure grows. And fiber-optic "
       "customers without tech support are the hottest segment."),
    img_cell(os.path.join(CHARTS, "01_churn_by_contract.png")),
    img_cell(os.path.join(CHARTS, "02_churn_by_tenure.png")),
    img_cell(os.path.join(CHARTS, "03_churn_by_service.png")),

    md("## 3. Modeling setup\n"
       "Stratified 80/20 split. Three models — logistic regression, random forest, "
       "gradient boosting — all with balanced class weights for the 26.5/73.5 "
       "imbalance, compared with 5-fold stratified CV on ROC-AUC."),
    code("""from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

num = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
cat = [c for c in X.columns if c not in num]
pre = ColumnTransformer([("num", StandardScaler(), num),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)"""),

    md("## 4. Results (test set, n=1,409)\n"
       "Logistic regression wins on ROC-AUC (0.841) and recall (78.3%) — the two "
       "metrics that matter when missing a churner costs more than a false alarm. "
       "The simpler model winning is a finding, not a failure."),
    img_cell(os.path.join(CHARTS, "04_roc_curves.png")),
    img_cell(os.path.join(CHARTS, "05_confusion_matrix.png")),

    md("## 5. Threshold tuning\n"
       "The default 0.5 threshold isn't sacred. For retention, recall matters more "
       "than precision — this curve shows exactly what each threshold buys."),
    img_cell(os.path.join(CHARTS, "06_threshold_tradeoff.png")),

    md("## 6. What drives churn?\n"
       "Permutation importance on the winning model: tenure dominates, then "
       "internet service, contract type, and monthly charges."),
    img_cell(os.path.join(CHARTS, "07_feature_importance.png")),

    md("## 7. Business readout\n"
       "- **High-risk segment (p ≥ 0.7): 1,783 customers (25.3% of base)** churning "
       "at **62.6% vs 26.5%** overall — ~$143k/month revenue at risk.\n"
       "- **Profile:** month-to-month + fiber optic + no tech support.\n"
       "- **Recommendation:** push contract upgrades and support add-ons at this "
       "segment inside the first 6 months; stop reporting churn as one blended number.\n\n"
       "Full pipeline in `churn.py`; metrics in `model_metrics.csv`."),
]

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python"}}, "cells": cells}

with open(os.path.join(HERE, "notebook.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
print("notebook.ipynb written with", len(cells), "cells")
