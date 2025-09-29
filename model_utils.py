import json, joblib, numpy as np, pandas as pd, shap
from pathlib import Path

ART = Path("artifacts")

MODELS = joblib.load(ART / "models.pkl")
CALS   = joblib.load(ART / "calibrators.pkl")
SCALER = joblib.load(ART / "scaler.pkl")
OOD    = joblib.load(ART / "ood.pkl")
CONF   = json.load(open(ART / "conformal.json"))
META   = json.load(open(ART / "feature_meta.json"))
STATS  = json.load(open(ART / "train_stats.json"))

FEATS = META["features"]
NUMS  = META["numeric_features"]
BINS  = META["binary_features"]
CATS  = META["categorical_features"]
CATEGORIES = META["categories"]
UNITS = META["units"]
RANGES = META["ranges"]
Q = float(CONF["q"])

# SHAP explainer on the first GBM model
_TREE_EXPLAINER = shap.TreeExplainer(MODELS[0])

def _prepare_df(patient_dict):
    """Cast to proper dtypes, order columns, and impute sensible defaults."""
    row = {k: patient_dict.get(k, None) for k in FEATS}
    df = pd.DataFrame([row])

    # numeric: coerce and fill with training medians
    for c in NUMS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            df[c] = df[c].fillna(STATS["numeric_median"][c])

    # binary expectations: coerce to {0,1}
    for c in BINS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).clip(0,1)

    # categorical: set to category with training categories; unseen → 'unknown' (if present) else first category
    for c in CATS:
        cats = CATEGORIES[c]
        df[c] = df[c].astype(str).str.strip().str.lower()
        df[c] = pd.Categorical(df[c], categories=cats)
        if df[c].isna().any():
            if "unknown" in cats:
                df[c] = df[c].cat.add_categories(["unknown"]).fillna("unknown")
            else:
                df[c] = df[c].fillna(cats[0])
        # Ensure dtype category (LightGBM reads it)
        df[c] = df[c].astype("category")

    # order columns
    df = df[FEATS]
    return df

def predict_with_uncertainty(patient):
    X = _prepare_df(patient)

    # Ensemble calibrated probability
    ps = np.stack([cal.predict_proba(X)[:,1] for cal in CALS], axis=0)
    p_mean = float(ps.mean())
    p_std  = float(ps.std())  # epistemic proxy

    # Conformal band
    lo = max(0.0, p_mean - Q); hi = min(1.0, p_mean + Q)

    # SHAP local drivers (for class=1)
    shap_vals = _TREE_EXPLAINER.shap_values(X)
    if isinstance(shap_vals, list):  # older shap returns [class0, class1]
        shap_arr = shap_vals[1]
    else:
        shap_arr = shap_vals
    contrib = sorted(
        [(f, float(shap_arr[0, i])) for i, f in enumerate(FEATS)],
        key=lambda t: abs(t[1]),
        reverse=True
    )[:5]

    # OOD using numeric features (scaled) → IsolationForest
    Xs = X[NUMS].copy()
    Xs_scaled = SCALER.transform(Xs)
    ood_score = OOD.decision_function(Xs_scaled)[0]  # lower → more OOD
    ood_flag = bool(ood_score < -0.1)  # threshold to tune

    return {
        "risk": p_mean,
        "uncertainty": {"lower": lo, "upper": hi, "epistemic_std": p_std},
        "drivers": contrib,
        "ood_flag": ood_flag
    }

def what_if(patient, tweaks):
    base = predict_with_uncertainty(patient)
    new_patient = {**patient, **tweaks}
    new = predict_with_uncertainty(new_patient)
    return {"before": base, "after": new}

def toggle_features(patient, removed_features):
    """Set removed numeric features to training median (simulate 'unknown').
       For binary/categorical removed features, set to benign/default."""
    p2 = patient.copy()
    for f in removed_features:
        if f in NUMS:
            p2[f] = STATS["numeric_median"][f]
        elif f in BINS:
            p2[f] = 0
        elif f in CATS:
            # choose 'unknown' if available else first category
            cats = CATEGORIES[f]
            p2[f] = "unknown" if "unknown" in cats else cats[0]
    return predict_with_uncertainty(p2)

def recommend_tests(patient, candidate_features):
    """
    Rank NUMERIC features by expected reduction of conformal band width.
    (Categoricals are skipped in this simple MVP.)
    """
    base = predict_with_uncertainty(patient)
    base_w = base["uncertainty"]["upper"] - base["uncertainty"]["lower"]
    if base_w <= 0:
        return {"base_width": base_w, "ranking": []}

    rng = np.random.default_rng(123)
    results = []
    for f in candidate_features:
        if f not in NUMS:
            continue
        mu = STATS["numeric_median"][f]
        # sample around range percentiles (closer to likely re-measurements)
        lo, hi = RANGES.get(f, [mu - 1.0, mu + 1.0])
        samples = rng.uniform(lo, hi, size=80)

        widths = []
        for s in samples:
            newp = {**patient, f: float(s)}
            w = predict_with_uncertainty(newp)["uncertainty"]
            widths.append(w["upper"] - w["lower"])
        exp_w = float(np.mean(widths))
        results.append({
            "feature": f,
            "expected_width": exp_w,
            "expected_reduction": float(max(0.0, base_w - exp_w))
        })

    results.sort(key=lambda d: d["expected_reduction"], reverse=True)
    return {"base_width": float(base_w), "ranking": results}

# Expose META to API
def get_meta():
    return {
        "features": FEATS,
        "units": UNITS,
        "ranges": RANGES,
        "numeric_features": NUMS,
        "binary_features": BINS,
        "categorical_features": CATS,
        "categories": CATEGORIES
    }
