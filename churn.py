"""Telecom customer churn prediction (supervised ML).

Business problem: predict which customers will churn so the retention team
can intervene *before* they leave — the classic ML use case in telecom.

Data: IBM Telco Customer Churn sample dataset (7,043 customers, 21 columns),
via the public mirror documented in the README.
Pipeline: clean -> EDA -> stratified split -> 3 models (class-imbalance
handled) -> threshold tuned for recall -> business readout.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score

HERE = "/home/hatch/workspace/data-analyst-projects/telecom-churn-prediction"
OUT = f"{HERE}/charts"
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

# ---------- 1. load + clean ----------
df = pd.read_csv(f"{HERE}/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"rows={len(df):,} churn_rate={df['Churn'].eq('Yes').mean():.1%} "
      f"blank TotalCharges={df['TotalCharges'].isna().sum()}")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]
num = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
cat = [c for c in X.columns if c not in num]

# ---------- 2. EDA charts ----------
# 2a. churn by contract
c = df.groupby("Contract")["Churn"].agg(["mean", "size"]).reset_index()
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(c["Contract"], c["mean"] * 100, color=["#c62828", "#2e7d32", "#1565c0"])
for i, r in c.iterrows():
    ax.text(i, r["mean"] * 100 + 1, f"{r['mean']:.0%}\n(n={r['size']:,})",
            ha="center", fontsize=9)
ax.set_title("Churn rate by contract type")
ax.set_ylabel("Churn rate (%)"); fig.tight_layout()
fig.savefig(f"{OUT}/01_churn_by_contract.png"); plt.close(fig)

# 2b. churn by tenure band
df["tenure_band"] = pd.cut(df["tenure"], [0, 6, 12, 24, 48, 73],
                           labels=["0-6m", "7-12m", "1-2y", "2-4y", "4y+"])
t = df.groupby("tenure_band", observed=True)["Churn"].mean()
fig, ax = plt.subplots(figsize=(7, 4.5))
t.plot.bar(ax=ax, color="#6a1b9a")
ax.set_title("Churn rate by tenure — churn is front-loaded")
ax.set_ylabel("Churn rate"); ax.set_xlabel("")
fig.tight_layout(); fig.savefig(f"{OUT}/02_churn_by_tenure.png"); plt.close(fig)

# 2c. churn by internet service x tech support
s = df.groupby(["InternetService", "TechSupport"])["Churn"].mean().unstack()
fig, ax = plt.subplots(figsize=(8, 4.5))
s.plot.bar(ax=ax, color=["#ef6c00", "#2e7d32"])
ax.set_title("Churn: internet service × tech support add-on")
ax.set_ylabel("Churn rate"); ax.set_xlabel(""); ax.legend(title="Tech support")
fig.tight_layout(); fig.savefig(f"{OUT}/03_churn_by_service.png"); plt.close(fig)

# ---------- 3. modeling ----------
pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])
models = {
    "Logistic regression": LogisticRegression(max_iter=2000,
                                              class_weight="balanced"),
    "Random forest": RandomForestClassifier(n_estimators=300, random_state=42,
                                            class_weight="balanced",
                                            n_jobs=-1),
    "Gradient boosting": HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=15,
        class_weight="balanced", random_state=42),
}
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"train={len(X_tr):,} test={len(X_te):,} "
      f"test churn={y_te.mean():.1%}")

rows, roc_data = [], {}
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
    pipe.fit(X_tr, y_tr)
    pr = pipe.predict(X_te)
    ps = pipe.predict_proba(X_te)[:, 1]
    rows.append({"model": name,
                 "cv_ROC_AUC": f"{cv_auc.mean():.3f}±{cv_auc.std():.3f}",
                 "accuracy": accuracy_score(y_te, pr),
                 "precision": precision_score(y_te, pr),
                 "recall": recall_score(y_te, pr),
                 "f1": f1_score(y_te, pr),
                 "ROC_AUC": roc_auc_score(y_te, y_te * 0 + ps)})
    fpr, tpr, _ = roc_curve(y_te, ps)
    roc_data[name] = (fpr, tpr, roc_auc_score(y_te, ps))
    print(f"{name:20s} acc={rows[-1]['accuracy']:.3f} "
          f"prec={rows[-1]['precision']:.3f} rec={rows[-1]['recall']:.3f} "
          f"f1={rows[-1]['f1']:.3f} auc={rows[-1]['ROC_AUC']:.3f}")
metrics = pd.DataFrame(rows)
metrics.to_csv(f"{HERE}/model_metrics.csv", index=False)

best_name = metrics.sort_values("ROC_AUC", ascending=False).iloc[0]["model"]
print("best by ROC-AUC:", best_name)
best = Pipeline([("pre", pre), ("clf", models[best_name])]).fit(X_tr, y_tr)

# 3a. ROC curves
fig, ax = plt.subplots(figsize=(7, 5.5))
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves — churn prediction (test set)")
ax.legend(fontsize=9); fig.tight_layout()
fig.savefig(f"{OUT}/04_roc_curves.png"); plt.close(fig)

# 3b. confusion matrix (best model)
cm = confusion_matrix(y_te, best.predict(X_te))
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred: stay", "Pred: churn"])
ax.set_yticklabels(["Actual: stay", "Actual: churn"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", fontsize=14)
ax.set_title(f"Confusion matrix — {best_name} (test set)")
fig.tight_layout(); fig.savefig(f"{OUT}/05_confusion_matrix.png"); plt.close(fig)

# 3c. precision-recall tradeoff by threshold
ps = best.predict_proba(X_te)[:, 1]
prec, rec, thr = precision_recall_curve(y_te, ps)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(thr, prec[:-1], label="Precision", color="#1565c0")
ax.plot(thr, rec[:-1], label="Recall", color="#c62828")
ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score")
ax.set_title("Threshold tuning: catch more churners vs fewer false alarms")
ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/06_threshold_tradeoff.png"); plt.close(fig)

# 3d. feature importance (permutation, best model)
from sklearn.inspection import permutation_importance
pi = permutation_importance(best, X_te, y_te, n_repeats=5, random_state=42,
                            scoring="roc_auc", n_jobs=-1)
feat_names = list(X.columns)  # permutation on the pipeline -> original columns
imp = pd.Series(pi.importances_mean, index=feat_names).sort_values().tail(12)
fig, ax = plt.subplots(figsize=(7, 5.5))
imp.plot.barh(ax=ax, color="#00838f")
ax.set_title(f"What drives churn? (permutation importance, {best_name})")
ax.set_xlabel("Drop in ROC-AUC when shuffled"); fig.tight_layout()
fig.savefig(f"{OUT}/07_feature_importance.png"); plt.close(fig)
imp.round(4).to_csv(f"{HERE}/feature_importance.csv")

# ---------- 4. business readout ----------
proba_all = best.predict_proba(X)[:, 1]
df["churn_proba"] = proba_all
high = df[df["churn_proba"] >= 0.7]
rev = high["MonthlyCharges"].sum()
print(f"\nhigh-risk customers (p>=0.7): {len(high):,} "
      f"({len(high) / len(df):.1%} of base)")
print(f"monthly revenue at risk: Rs {rev:,.0f} (charges as-is from dataset)")
print(f"their churn rate: {high['Churn'].mean():.1%} vs base {y.mean():.1%}")
print("\nhigh-risk profile:")
print(high[["Contract", "InternetService", "TechSupport"]].apply(
    lambda s: s.value_counts().head(2)).to_string())
print("\ncharts written to", OUT)
