import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import lightgbm as lgb

DATA_PATH = "data/diabetes_prediction_dataset.csv"
ART = Path("artifacts"); ART.mkdir(exist_ok=True, parents=True)

# ---------- 1) Load & clean ----------
df = pd.read_csv(DATA_PATH)

# Expected columns
target_col = "diabetes"
raw_feature_cols = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"
]

# Basic cleaning / typing
df = df[raw_feature_cols + [target_col]].copy()
# Standardize categorical values
df["gender"] = df["gender"].str.strip().str.lower()
df["smoking_history"] = df["smoking_history"].astype(str).str.strip().str.lower()

# Map binary ints
for b in ["hypertension", "heart_disease", target_col]:
    df[b] = df[b].astype(int)

# Cast categoricals (LightGBM can use pandas category dtype)
cat_cols = ["gender", "smoking_history"]
for c in cat_cols:
    df[c] = df[c].astype("category")

num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]

# Handle missing values (simple strategy for MVP)
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

# For categoricals, fill missing with most frequent
for c in cat_cols:
    if df[c].isna().any():
        mode = df[c].mode(dropna=True)
        df[c] = df[c].cat.add_categories(["unknown"]).fillna("unknown").astype("category")
        if not mode.empty:
            df.loc[df[c].isna(), c] = mode.iloc[0]

X = df[raw_feature_cols].copy()
y = df[target_col].astype(int)

# ---------- 2) Split (train/val/test) ----------
X_tr, X_temp, y_tr, y_temp = train_test_split(
    X, y, test_size=0.40, stratify=y, random_state=42
)
X_val, X_te, y_val, y_te = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# Keep category metadata for inference
cat_categories = {c: list(X_tr[c].cat.categories) for c in cat_cols}

# ---------- 3) Train deep ensemble (LightGBM) ----------
def fit_lgbm(Xdf, y, seed):
    # LightGBM can ingest pandas categoricals directly
    clf = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed
    )
    clf.fit(Xdf, y, categorical_feature=cat_cols)
    return clf

seeds = [11, 22, 33, 44, 55]
models = [fit_lgbm(X_tr, y_tr, s) for s in seeds]

# ---------- 4) Calibrate each model on validation ----------
calibrators = []
for m in models:
    cal = CalibratedClassifierCV(m, method="isotonic", cv="prefit")
    cal.fit(X_val, y_val)
    calibrators.append(cal)

p_val = np.mean([cal.predict_proba(X_val)[:, 1] for cal in calibrators], axis=0)
print("Brier (val):", brier_score_loss(y_val, p_val))

# ---------- 5) Conformal band (split) ----------
# Nonconformity = |y - p|
alpha = 0.10  # target ~90% coverage
val_scores = np.abs(y_val.values - p_val)
q = float(np.quantile(val_scores, 1 - alpha))

# ---------- 6) Train OOD detector on TRAIN numeric features ----------
# Use StandardScaler on numerical columns only
scaler = StandardScaler().fit(X_tr[num_cols])
Xtr_num_scaled = scaler.transform(X_tr[num_cols])

ood = IsolationForest(
    n_estimators=300,
    contamination=0.02,  # tune if needed
    random_state=7
)
ood.fit(Xtr_num_scaled)

# ---------- 7) Feature metadata for UI ----------
# Ranges from robust percentiles (2nd–98th) to drive sliders
ranges = {}
for c in num_cols:
    lo, hi = np.percentile(X_tr[c], [2, 98])
    # ensure valid range (avoid equal bounds)
    if hi <= lo:
        hi = lo + 1.0
    ranges[c] = [float(lo), float(hi)]

# Binary indicators and categoricals get suggested sets, not ranges
units = {
    "age": "years",
    "bmi": "kg/m²",
    "HbA1c_level": "%",
    "blood_glucose_level": "mg/dL",  # adjust if your units differ
    "hypertension": "0/1",
    "heart_disease": "0/1",
}
feature_meta = {
    "features": raw_feature_cols,
    "numeric_features": num_cols,
    "binary_features": ["hypertension", "heart_disease"],
    "categorical_features": cat_cols,
    "categories": cat_categories,     # for casting at inference
    "units": units,
    "ranges": ranges
}

# ---------- 8) Persist artifacts ----------
joblib.dump(models, ART / "models.pkl")
joblib.dump(calibrators, ART / "calibrators.pkl")
joblib.dump(scaler, ART / "scaler.pkl")
joblib.dump(ood, ART / "ood.pkl")
json.dump({"alpha": alpha, "q": q}, open(ART / "conformal.json", "w"))
json.dump(feature_meta, open(ART / "feature_meta.json", "w"))

# Train stats for simple imputation in inference
train_stats = {
    "numeric_median": {c: float(X_tr[c].median()) for c in num_cols}
}
json.dump(train_stats, open(ART / "train_stats.json", "w"))

print("Saved artifacts to ./artifacts")
