import ctypes.util
import json
import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import lightgbm as lgb

# ---------- macOS libomp guard ----------
if sys.platform == "darwin" and not ctypes.util.find_library("omp"):
    raise OSError(
        "LightGBM requires the libomp runtime on macOS. Install it via `brew install libomp` "
        "before running training."
    )

# ---------- Paths ----------
DATA_PATH = "data/mental_health_data_realistic.csv"
ART = Path("artifacts"); ART.mkdir(exist_ok=True, parents=True)

print("="*60)
print("MENTAL HEALTH RISK MODEL TRAINING")
print("="*60)

# ---------- 1) Load & clean ----------
print("\n[1/8] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Loaded {len(df)} records")

# Target column
target_col = "Mental_Health_Condition"

# Feature columns based on the dataset description
# NOTE: Excluding "Severity" as it's directly related to the target (data leakage)
feature_cols = [
    # Demographics
    "Age", "Gender", "Occupation", "Country",
    # Mental Health History
    "Consultation_History",
    # Lifestyle - Numeric
    "Stress_Level", "Sleep_Hours", "Work_Hours", 
    "Physical_Activity_Hours", "Social_Media_Usage",
    # Lifestyle - Categorical
    "Diet_Quality", "Smoking_Habit", "Alcohol_Consumption",
    # Treatment
    "Medication_Usage"
]

# Check which columns exist
available_cols = [c for c in feature_cols if c in df.columns]
missing_cols = [c for c in feature_cols if c not in df.columns]

if missing_cols:
    print(f"   Warning: Missing columns: {missing_cols}")
    print(f"   Using available columns: {available_cols}")
    feature_cols = available_cols

# Subset data
df = df[feature_cols + [target_col]].copy()

print("\n[2/8] Preprocessing data...")

# Define categorical and numeric columns
cat_cols = [
    "Gender", "Occupation", "Country", 
    "Consultation_History", "Severity",
    "Diet_Quality", "Smoking_Habit", 
    "Alcohol_Consumption", "Medication_Usage"
]
# Only keep categorical columns that exist
cat_cols = [c for c in cat_cols if c in df.columns]

num_cols = [
    "Age", "Stress_Level", "Sleep_Hours",
    "Work_Hours", "Physical_Activity_Hours", "Social_Media_Usage"
]
# Only keep numeric columns that exist
num_cols = [c for c in num_cols if c in df.columns]

# Clean categorical columns - normalize strings
for c in cat_cols:
    df[c] = df[c].astype(str).str.strip().str.lower()
    # Handle missing values
    df[c] = df[c].replace(['nan', 'none', ''], 'unknown')

# Clean target - handle Yes/No or 1/0
df[target_col] = df[target_col].astype(str).str.strip().str.lower()
if df[target_col].isin(['yes', 'no']).any():
    df[target_col] = df[target_col].map({"yes": 1, "no": 0})
else:
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df[target_col] = df[target_col].fillna(0).astype(int)

# Convert to proper dtypes
for c in cat_cols:
    df[c] = df[c].astype("category")

# Numeric columns - coerce and impute
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    median_val = df[c].median()
    if pd.isna(median_val):
        median_val = 0.0
    df[c] = df[c].fillna(median_val)

# Check class balance
print(f"   Class distribution:")
print(df[target_col].value_counts())
print(f"   Positive rate: {df[target_col].mean():.2%}")

X = df[feature_cols].copy()
y = df[target_col].astype(int)

# ---------- 2) Split (train/val/test) ----------
print("\n[3/8] Splitting data...")

# Check if we have enough samples for stratification
if y.nunique() < 2 or y.value_counts().min() < 10:
    print("   Warning: Insufficient samples for stratification, using random split")
    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )
else:
    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

print(f"   Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_te)}")

# Store category metadata for inference
cat_categories = {c: list(X_tr[c].cat.categories) for c in cat_cols}

# ---------- 3) Train ensemble (LightGBM) ----------
print("\n[4/8] Training LightGBM ensemble...")

def fit_lgbm(Xdf, y, seed):
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        verbose=-1
    )
    clf.fit(
        Xdf, y, 
        categorical_feature=cat_cols,
        eval_set=[(Xdf, y)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    return clf

seeds = [11, 22, 33, 44, 55]
models = []
for i, s in enumerate(seeds, 1):
    print(f"   Training model {i}/{len(seeds)}...", end=" ")
    m = fit_lgbm(X_tr, y_tr, s)
    models.append(m)
    # Quick validation
    train_pred = m.predict_proba(X_tr)[:, 1]
    train_auc = roc_auc_score(y_tr, train_pred)
    print(f"Train AUC: {train_auc:.3f}")

# ---------- 4) Calibrate each model on validation ----------
print("\n[5/8] Calibrating models...")
calibrators = []
p_val_stack = []

for i, m in enumerate(models, 1):
    cal = CalibratedClassifierCV(m, method="isotonic", cv="prefit")
    cal.fit(X_val, y_val)
    calibrators.append(cal)
    p_val = cal.predict_proba(X_val)[:, 1]
    p_val_stack.append(p_val)
    print(f"   Model {i} - Val AUC: {roc_auc_score(y_val, p_val):.3f}")

p_val_stack = np.stack(p_val_stack, axis=0)
p_val_mean = np.mean(p_val_stack, axis=0)
ens_std_val = np.std(p_val_stack, axis=0)
aleatoric_val = np.mean(p_val_stack * (1 - p_val_stack), axis=0)

print(f"\n   Ensemble Val AUC: {roc_auc_score(y_val, p_val_mean):.3f}")
print(f"   Ensemble Val Brier: {brier_score_loss(y_val, p_val_mean):.3f}")

# Test set evaluation
p_test_stack = []
for cal in calibrators:
    p_test_stack.append(cal.predict_proba(X_te)[:, 1])
p_test_mean = np.mean(p_test_stack, axis=0)
print(f"   Ensemble Test AUC: {roc_auc_score(y_te, p_test_mean):.3f}")
print(f"   Ensemble Test Brier: {brier_score_loss(y_te, p_test_mean):.3f}")

# ---------- 5) Conformal prediction intervals ----------
print("\n[6/8] Computing conformal intervals...")

alpha = 0.10  # 90% coverage target
eps = 1e-6

# Scale by combined uncertainty
total_scale = np.sqrt(np.clip(aleatoric_val + ens_std_val**2, 0.0, None) + eps)
val_scores = np.abs(y_val.values - p_val_mean) / total_scale
q = float(np.quantile(val_scores, 1 - alpha))

print(f"   Conformal quantile (q): {q:.3f}")
print(f"   Target coverage: {(1-alpha)*100:.0f}%")

# Evaluate coverage on test set
p_test_stack_arr = np.stack(p_test_stack, axis=0)
test_std = np.std(p_test_stack_arr, axis=0)
test_aleatoric = np.mean(p_test_stack_arr * (1 - p_test_stack_arr), axis=0)
test_total_scale = np.sqrt(np.clip(test_aleatoric + test_std**2, 0.0, None) + eps)
test_delta = q * test_total_scale
test_lo = np.clip(p_test_mean - test_delta, 0, 1)
test_hi = np.clip(p_test_mean + test_delta, 0, 1)
test_coverage = np.mean((y_te.values >= test_lo) & (y_te.values <= test_hi))
print(f"   Actual test coverage: {test_coverage*100:.1f}%")
print(f"   Average interval width: {np.mean(test_hi - test_lo):.3f}")

# ---------- 6) Train OOD detector ----------
print("\n[7/8] Training out-of-distribution detector...")

scaler = StandardScaler().fit(X_tr[num_cols])
Xtr_num_scaled = scaler.transform(X_tr[num_cols])

ood = IsolationForest(
    n_estimators=300,
    contamination=0.05,
    random_state=7,
    verbose=0
)
ood.fit(Xtr_num_scaled)
print("   OOD detector trained")

# ---------- 7) Feature metadata ----------
print("\n[8/8] Saving artifacts...")

# Compute ranges for numeric features
ranges = {}
for c in num_cols:
    vals = X_tr[c].dropna()
    if len(vals) > 0:
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        ranges[c] = [float(lo), float(hi)]
    else:
        ranges[c] = [0.0, 1.0]

units = {
    "Age": "years",
    "Stress_Level": "0-10 scale",
    "Sleep_Hours": "hours/night",
    "Work_Hours": "hours/week",
    "Physical_Activity_Hours": "hours/week",
    "Social_Media_Usage": "hours/day",
}

feature_meta = {
    "features": feature_cols,
    "numeric_features": num_cols,
    "binary_features": [],
    "categorical_features": cat_cols,
    "categories": cat_categories,
    "units": units,
    "ranges": ranges
}

# Training statistics
train_stats = {
    "numeric_median": {c: float(X_tr[c].median()) for c in num_cols},
    "categorical_modes": {c: str(X_tr[c].mode()[0]) if len(X_tr[c].mode()) > 0 else 'unknown' 
                          for c in cat_cols}
}

# ---------- 8) Save everything ----------
joblib.dump(models, ART / "models.pkl")
joblib.dump(calibrators, ART / "calibrators.pkl")
joblib.dump(scaler, ART / "scaler.pkl")
joblib.dump(ood, ART / "ood.pkl")

json.dump({
    "alpha": alpha, 
    "q": q,
    "epsilon": eps
}, open(ART / "conformal.json", "w"), indent=2)

json.dump(feature_meta, open(ART / "feature_meta.json", "w"), indent=2)
json.dump(train_stats, open(ART / "train_stats.json", "w"), indent=2)

# Save performance metrics
metrics = {
    "val_auc": float(roc_auc_score(y_val, p_val_mean)),
    "val_brier": float(brier_score_loss(y_val, p_val_mean)),
    "test_auc": float(roc_auc_score(y_te, p_test_mean)),
    "test_brier": float(brier_score_loss(y_te, p_test_mean)),
    "conformal_coverage": float(test_coverage),
    "avg_interval_width": float(np.mean(test_hi - test_lo))
}
json.dump(metrics, open(ART / "metrics.json", "w"), indent=2)

print("\n" + "="*60)
print("✅ TRAINING COMPLETE")
print("="*60)
print(f"\nArtifacts saved to: {ART.absolute()}")
print(f"\nModel Performance:")
print(f"  Validation AUC:  {metrics['val_auc']:.3f}")
print(f"  Test AUC:        {metrics['test_auc']:.3f}")
print(f"  Conformal Coverage: {metrics['conformal_coverage']*100:.1f}%")
print(f"  Avg Interval Width: {metrics['avg_interval_width']:.3f}")
print("\n" + "="*60)
