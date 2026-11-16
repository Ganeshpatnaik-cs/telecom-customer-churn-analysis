"""
Generate a synthetic dataset that follows the schema of the public
"Telco Customer Churn" dataset (IBM sample data on Kaggle).

Why synthetic? It lets this project run anywhere with no download.
To use the real data, download WA_Fn-UseC_-Telco-Customer-Churn.csv from Kaggle
and save it as data/telco_churn.csv -- churn_analysis.py works on both.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 7043

def pick(options, probs, size=N):
    return rng.choice(options, size=size, p=probs)

df = pd.DataFrame({
    "customerID": [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(N)],
    "gender": pick(["Male", "Female"], [0.5, 0.5]),
    "SeniorCitizen": pick([0, 1], [0.84, 0.16]),
    "Partner": pick(["Yes", "No"], [0.48, 0.52]),
    "Dependents": pick(["Yes", "No"], [0.30, 0.70]),
})

df["Contract"] = pick(["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24])

# Tenure depends on contract type
tenure = np.where(
    df["Contract"] == "Month-to-month", rng.gamma(1.6, 12, N),
    np.where(df["Contract"] == "One year", rng.normal(40, 16, N), rng.normal(56, 14, N)),
)
df["tenure"] = np.clip(tenure.round(), 1, 72).astype(int)

df["PhoneService"] = pick(["Yes", "No"], [0.90, 0.10])
df["MultipleLines"] = np.where(
    df["PhoneService"] == "No", "No phone service", pick(["Yes", "No"], [0.47, 0.53])
)
df["InternetService"] = pick(["Fiber optic", "DSL", "No"], [0.44, 0.34, 0.22])

def internet_addon(p_yes):
    return np.where(df["InternetService"] == "No", "No internet service", pick(["Yes", "No"], [p_yes, 1 - p_yes]))

df["OnlineSecurity"] = internet_addon(0.29)
df["OnlineBackup"] = internet_addon(0.34)
df["DeviceProtection"] = internet_addon(0.34)
df["TechSupport"] = internet_addon(0.29)
df["StreamingTV"] = internet_addon(0.38)
df["StreamingMovies"] = internet_addon(0.39)

df["PaperlessBilling"] = pick(["Yes", "No"], [0.59, 0.41])
df["PaymentMethod"] = pick(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    [0.34, 0.23, 0.22, 0.21],
)

# Monthly charges built from services
charge = np.full(N, 20.0)
charge += np.where(df["PhoneService"] == "Yes", 10, 0)
charge += np.where(df["MultipleLines"] == "Yes", 5, 0)
charge += np.where(df["InternetService"] == "DSL", 25, 0)
charge += np.where(df["InternetService"] == "Fiber optic", 50, 0)
for col, amt in [("OnlineSecurity", 5), ("OnlineBackup", 5), ("DeviceProtection", 5),
                 ("TechSupport", 5), ("StreamingTV", 8), ("StreamingMovies", 8)]:
    charge += np.where(df[col] == "Yes", amt, 0)
charge += rng.normal(0, 3, N)
df["MonthlyCharges"] = np.clip(charge, 18.25, 118.75).round(2)
df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"] * rng.normal(1.0, 0.03, N)).round(2)

# Churn: logistic model with realistic drivers
z = -1.95
z += np.where(df["Contract"] == "Month-to-month", 1.3, 0)
z += np.where(df["Contract"] == "Two year", -1.4, 0)
z += -0.035 * df["tenure"]
z += np.where(df["InternetService"] == "Fiber optic", 0.7, 0)
z += np.where(df["InternetService"] == "No", -0.8, 0)
z += np.where(df["TechSupport"] == "No", 0.4, 0)
z += np.where(df["OnlineSecurity"] == "No", 0.35, 0)
z += np.where(df["PaymentMethod"] == "Electronic check", 0.5, 0)
z += np.where(df["PaperlessBilling"] == "Yes", 0.2, 0)
z += np.where(df["SeniorCitizen"] == 1, 0.3, 0)
z += np.where(df["Dependents"] == "Yes", -0.25, 0)
z += 0.008 * (df["MonthlyCharges"] - 65)
p = 1 / (1 + np.exp(-z))
df["Churn"] = np.where(rng.random(N) < p, "Yes", "No")

# Mimic the real dataset's quirk: a few blank TotalCharges for brand-new customers
blank_idx = rng.choice(df.index[df["tenure"] == 1], size=11, replace=False)
df["TotalCharges"] = df["TotalCharges"].astype(object)
df.loc[blank_idx, "TotalCharges"] = " "

df.to_csv("data/telco_churn.csv", index=False)
print(f"Wrote data/telco_churn.csv  rows={len(df)}  churn_rate={(df.Churn=='Yes').mean():.1%}")
