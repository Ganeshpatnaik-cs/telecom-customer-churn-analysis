"""
Customer Churn Analysis and Prediction
Works with the Telco Customer Churn schema (synthetic or the real Kaggle file).

Run:  python churn_analysis.py [path_to_csv]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/telco_churn.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"No": "#4C78A8", "Yes": "#E45756"}

# ---------------------------------------------------------------- 1. Load + clean
df = pd.read_csv(DATA)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_missing = df["TotalCharges"].isna().sum()
df["TotalCharges"] = df["TotalCharges"].fillna(0)  # blanks are brand-new customers
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
print(f"Fixed {n_missing} blank TotalCharges values | duplicates: {df.duplicated('customerID').sum()}")

overall = df["ChurnFlag"].mean()
print(f"Overall churn rate: {overall:.1%}")

# ---------------------------------------------------------------- 2. EDA
lines = [f"Overall churn rate: {overall:.1%}  ({df['ChurnFlag'].sum():,} of {len(df):,} customers)\n"]

def churn_by(col):
    t = df.groupby(col)["ChurnFlag"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    t["mean"] = (t["mean"] * 100).round(1)
    return t.rename(columns={"mean": "churn_%", "count": "customers"})

for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity", "SeniorCitizen"]:
    t = churn_by(col)
    lines.append(f"Churn by {col}:\n{t.to_string()}\n")

df["TenureBand"] = pd.cut(df["tenure"], [0, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
t = df.groupby("TenureBand", observed=True)["ChurnFlag"].mean().mul(100).round(1)
lines.append(f"Churn % by tenure band (months):\n{t.to_string()}\n")

# Chart 1: churn by contract
fig, ax = plt.subplots(figsize=(6, 4))
c = churn_by("Contract")["churn_%"]
ax.bar(c.index, c.values, color="#E45756")
ax.axhline(overall * 100, ls="--", color="grey", lw=1)
ax.text(2.45, overall * 100 + 1, f"avg {overall:.0%}", ha="right", color="grey", fontsize=8)
for i, v in enumerate(c.values):
    ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
ax.set(title="Churn rate by contract type", ylabel="Churn %")
fig.tight_layout(); fig.savefig(OUT / "01_churn_by_contract.png"); plt.close(fig)

# Chart 2: churn by tenure band
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(t.index.astype(str), t.values, color="#E45756")
for i, v in enumerate(t.values):
    ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
ax.set(title="Churn rate by customer tenure", xlabel="Tenure (months)", ylabel="Churn %")
fig.tight_layout(); fig.savefig(OUT / "02_churn_by_tenure.png"); plt.close(fig)

# Chart 3: monthly charges distribution
fig, ax = plt.subplots(figsize=(6, 4))
for k in ["No", "Yes"]:
    ax.hist(df.loc[df["Churn"] == k, "MonthlyCharges"], bins=30, alpha=0.65,
            color=COLORS[k], label=f"Churn = {k}")
ax.set(title="Monthly charges: churned vs retained", xlabel="Monthly charges", ylabel="Customers")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "03_monthly_charges.png"); plt.close(fig)

# Chart 4: churn by internet service and payment method
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col in zip(axes, ["InternetService", "PaymentMethod"]):
    s = churn_by(col)["churn_%"].sort_values()
    ax.barh(s.index, s.values, color="#E45756")
    for i, v in enumerate(s.values):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=9)
    ax.set(title=f"Churn rate by {col}", xlabel="Churn %")
fig.tight_layout(); fig.savefig(OUT / "04_churn_by_service_payment.png"); plt.close(fig)

# ---------------------------------------------------------------- 3. Modeling
features = df.drop(columns=["customerID", "Churn", "ChurnFlag", "TenureBand"])
target = df["ChurnFlag"]
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [c for c in features.columns if c not in num_cols]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, stratify=target, random_state=42)

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            class_weight="balanced", random_state=42, n_jobs=-1),
}

lines.append("=" * 60 + "\nMODEL RESULTS (20% held-out test set)\n" + "=" * 60)
fitted, probs = {}, {}
fig, ax = plt.subplots(figsize=(5.5, 5))
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    auc = roc_auc_score(y_test, p)
    fitted[name], probs[name] = pipe, p
    lines.append(f"\n{name}  |  ROC-AUC = {auc:.3f}\n"
                 f"{classification_report(y_test, pred, target_names=['Stayed', 'Churned'], digits=3)}"
                 f"Confusion matrix [[TN FP],[FN TP]]:\n{confusion_matrix(y_test, pred)}")
    fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "05_roc_curves.png"); plt.close(fig)

# Feature importance from the logistic regression (standardized coefficients)
best = fitted["Logistic Regression"]
names = best.named_steps["pre"].get_feature_names_out()
coefs = pd.Series(best.named_steps["clf"].coef_[0], index=names).sort_values()
top = pd.concat([coefs.head(8), coefs.tail(8)])
top.index = [i.replace("num__", "").replace("cat__", "") for i in top.index]
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.barh(top.index, top.values, color=["#4C78A8" if v < 0 else "#E45756" for v in top.values])
ax.axvline(0, color="black", lw=0.8)
ax.set(title="Top churn drivers (logistic regression coefficients)",
       xlabel="<- lowers churn risk | raises churn risk ->")
fig.tight_layout(); fig.savefig(OUT / "06_feature_importance.png"); plt.close(fig)

lines.append("\nTop factors RAISING churn risk (logistic regression):")
lines.append(coefs.tail(6).iloc[::-1].rename(lambda s: s.replace("cat__", "").replace("num__", "")).round(3).to_string())
lines.append("\nTop factors LOWERING churn risk:")
lines.append(coefs.head(6).rename(lambda s: s.replace("cat__", "").replace("num__", "")).round(3).to_string())

# ---------------------------------------------------------------- 4. Export for Power BI / Tableau
df["ChurnProbability"] = fitted["Logistic Regression"].predict_proba(features)[:, 1].round(4)
df["RiskSegment"] = pd.cut(df["ChurnProbability"], [-0.01, 0.3, 0.6, 1.0],
                           labels=["Low", "Medium", "High"])
df.drop(columns=["ChurnFlag"]).to_csv(OUT / "churn_scored_for_dashboard.csv", index=False)

hi = df[df["RiskSegment"] == "High"]
lines.append(f"\nHigh-risk customers: {len(hi):,} "
             f"| their monthly revenue: {hi['MonthlyCharges'].sum():,.0f} "
             f"| actual churn rate in this group: {hi['ChurnFlag'].mean():.1%}")

(OUT / "results_summary.txt").write_text("\n".join(lines))
print("\n".join(lines))
print(f"\nSaved charts, results_summary.txt and churn_scored_for_dashboard.csv to {OUT}/")
