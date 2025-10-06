import ctypes.util
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------- macOS libomp guard ----------
if sys.platform == "darwin" and not ctypes.util.find_library("omp"):
    raise OSError(
        "LightGBM requires the libomp runtime on macOS. Install it via `brew install libomp` "
        "and restart the app."
    )

import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import IsolationForest
import shap

st.set_page_config(page_title="Mental Health Risk + Uncertainty", layout="wide")
st.title("🧠 Mental Health Condition Risk + Uncertainty")

# ----------------------------
# App constants (dataset schema)
# ----------------------------
EXPECTED_COLUMNS = [
    "Gender","Age","Occupation","Country","Consultation_History",
    "Stress_Level","Sleep_Hours","Work_Hours","Physical_Activity_Hours",
    "Social_Media_Usage","Diet_Quality","Smoking_Habit","Alcohol_Consumption",
    "Medication_Usage","Mental_Health_Condition"
]
TARGET_COL = "Mental_Health_Condition"
CAT_COLS = [
    "Gender","Occupation","Country","Consultation_History",
    "Diet_Quality","Smoking_Habit","Alcohol_Consumption","Medication_Usage"
]
BIN_COLS: List[str] = []   # we keep yes/no as categories in this setup
NUM_COLS = ["Age","Stress_Level","Sleep_Hours","Work_Hours",
            "Physical_Activity_Hours","Social_Media_Usage"]

UNITS = {
    "Age": "years",
    "Stress_Level": "0–10",
    "Sleep_Hours": "hours",
    "Work_Hours": "hours/day",
    "Physical_Activity_Hours": "hours/week",
    "Social_Media_Usage": "hours/day",
}

# ----------------------------
# Safety helpers
# ----------------------------
def _safe_numeric_bounds(series: pd.Series) -> tuple[float, float]:
    """Return robust (lo, hi) for a numeric-like series, guaranteed finite and lo < hi."""
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return 0.0, 1.0
    lo = float(np.nanpercentile(s, 2))
    hi = float(np.nanpercentile(s, 98))
    # fallback to median if needed
    if not np.isfinite(lo):
        lo = float(np.nanmedian(s))
    if not np.isfinite(hi):
        hi = float(np.nanmedian(s))
    # ensure spread
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        med = float(np.nanmedian(s))
        lo, hi = med - 1.0, med + 1.0
    # cap extreme spans
    span = hi - lo
    if not np.isfinite(span) or span <= 0 or span > 1e12:
        med = float(np.nanmedian(s))
        lo, hi = med - 1.0, med + 1.0
    return float(lo), float(hi)

def _sanitize_bounds(lo: float, hi: float, fallback_med: float) -> tuple[float, float]:
    """Sanitize (lo,hi) just before sampling."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = fallback_med - 1.0, fallback_med + 1.0
    span = hi - lo
    if not np.isfinite(span) or span <= 0 or span > 1e12:
        lo, hi = fallback_med - 1.0, fallback_med + 1.0
    return float(lo), float(hi)

def prob_of_one(cal: CalibratedClassifierCV, X: pd.DataFrame) -> np.ndarray:
    """
    Robustly return P(y=1) from a calibrated classifier.
    Handles single-class calibration edge cases.
    """
    p = cal.predict_proba(X)
    if p.ndim == 1:
        return p
    if p.shape[1] == 1:
        classes = getattr(cal, "classes_", np.array([0]))
        c = int(classes[0])
        return p[:, 0] if c == 1 else 1.0 - p[:, 0]
    classes = getattr(cal, "classes_", np.array([0, 1]))
    if 1 in classes:
        j = int(np.where(classes == 1)[0][0])
        return p[:, j]
    return p[:, -1]

# ----------------------------
# Data loading
# ----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader(
    "Upload mental_health_data.csv", type=["csv"]
)

@st.cache_data(show_spinner=True)
def load_default_sample() -> pd.DataFrame:
    # tiny synthetic fallback (keeps app runnable if no upload)
    rng = np.random.default_rng(7)
    n = 1500
    df = pd.DataFrame({
        "Gender": rng.choice(["male","female","non-binary","prefer not to say"], size=n),
        "Age": rng.integers(18, 66, size=n),
        "Occupation": rng.choice(["it","healthcare","education","engineering","finance","other"], size=n),
        "Country": rng.choice(["canada","usa","india","germany","uk","australia","other"], size=n),
        "Consultation_History": rng.choice(["yes","no","first-time","irregular"], size=n),
        "Stress_Level": rng.integers(0, 11, size=n),
        "Sleep_Hours": rng.normal(7.0, 1.4, size=n).clip(3, 12),
        "Work_Hours": rng.integers(4, 13, size=n),
        "Physical_Activity_Hours": rng.integers(0, 11, size=n),
        "Social_Media_Usage": rng.uniform(0.5, 6.0, size=n),
        "Diet_Quality": rng.choice(["healthy","average","unhealthy"], size=n),
        "Smoking_Habit": rng.choice(["non-smoker","occasional smoker","regular smoker","heavy smoker"], size=n),
        "Alcohol_Consumption": rng.choice(["non-drinker","social drinker","regular drinker","heavy drinker"], size=n),
        "Medication_Usage": rng.choice(["yes","no"], size=n),
    })
    # synthetic label: higher stress/low sleep -> higher risk
    logit = (
        -2.2
        + 0.18*(df["Stress_Level"])
        + 0.12*(12 - np.clip(df["Sleep_Hours"], 0, 12))
        + 0.03*(df["Work_Hours"] - 8)
        + 0.02*(df["Social_Media_Usage"] - 2)
        + 0.2*(df["Medication_Usage"].astype(str).str.lower().eq("yes")).astype(int)
        + 0.1*(df["Consultation_History"].astype(str).str.lower().isin(["yes","irregular"])).astype(int)
    )
    p = 1 / (1 + np.exp(-logit))
    df["Mental_Health_Condition"] = (rng.uniform(0, 1, size=n) < p).astype(int)
    return df

if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Dataset loaded from upload.")
else:
    df = load_default_sample()
    st.sidebar.info("No file uploaded — using a small synthetic sample.")

# Validate/align columns
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}\n\nExpected: {EXPECTED_COLUMNS}")
    st.stop()

df = df[EXPECTED_COLUMNS].copy()

# Clean / cast
for c in CAT_COLS + [TARGET_COL]:
    df[c] = df[c].astype(str).str.strip().str.lower()

# Ensure target is 0/1 (accepts 'yes'/'no' or already numeric)
if df[TARGET_COL].dtype.kind in "iufc":
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)
else:
    df[TARGET_COL] = df[TARGET_COL].map({"yes":1, "no":0}).fillna(0).astype(int)

# Numerics: coerce + median impute
for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

# set categoricals
for c in CAT_COLS:
    df[c] = df[c].astype("category")

# ----------------------------
# Train / cache the model stack
# ----------------------------
@dataclass
class Artifacts:
    models: List[lgb.LGBMClassifier]
    calibrators: List[CalibratedClassifierCV]
    conformal_q: float
    scaler: StandardScaler
    ood: IsolationForest
    cat_categories: Dict[str, List[str]]
    ranges: Dict[str, List[float]]
    explainer: shap.TreeExplainer  # for local explanations

@st.cache_resource(show_spinner=True)
def train_all(df: pd.DataFrame) -> Artifacts:
    X = df[[c for c in EXPECTED_COLUMNS if c != TARGET_COL]].copy()
    y = df[TARGET_COL].astype(int)

    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=42)
    X_val, X_te,  y_val, y_te  = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    # guard against single-class validation by retrying a few seeds
    if (y_tr.nunique() < 2) or (y_val.nunique() < 2):
        for rs in range(43, 63):
            X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=rs)
            X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=rs)
            if (y_tr.nunique() == 2) and (y_val.nunique() == 2):
                break

    cat_categories = {c: list(X_tr[c].cat.categories) for c in CAT_COLS}

    def fit_lgbm(Xdf, y, seed):
        clf = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.03,
            num_leaves=63, subsample=0.9, colsample_bytree=0.9,
            random_state=seed
        )
        clf.fit(Xdf, y, categorical_feature=CAT_COLS)
        return clf

    seeds = [11,22,33,44,55]
    models = [fit_lgbm(X_tr, y_tr, s) for s in seeds]

    calibrators = []
    for m in models:
        cal = CalibratedClassifierCV(m, method="isotonic", cv="prefit")
        cal.fit(X_val, y_val)
        calibrators.append(cal)

    # robust proba extraction
    p_val = np.mean([prob_of_one(cal, X_val) for cal in calibrators], axis=0)
    _brier = brier_score_loss(y_val, p_val)

    # conformal split: nonconformity |y - p|
    alpha = 0.10
    val_scores = np.abs(y_val.values - p_val)
    q = float(np.quantile(val_scores, 1 - alpha))

    # OOD on numeric space
    scaler = StandardScaler().fit(X_tr[NUM_COLS])
    Xtr_scaled = scaler.transform(X_tr[NUM_COLS])
    ood = IsolationForest(n_estimators=300, contamination=0.02, random_state=7)
    ood.fit(Xtr_scaled)

    # numeric slider ranges (2nd–98th) with safety guards
    ranges = {}
    for c in NUM_COLS:
        lo, hi = _safe_numeric_bounds(X_tr[c])
        ranges[c] = [lo, hi]

    # SHAP (TreeExplainer on first model)
    explainer = shap.TreeExplainer(models[0])

    st.sidebar.caption(
        f"Validation Brier score: {round(float(_brier), 4)} | Conformal q: {round(float(q), 3)}"
    )
    return Artifacts(models, calibrators, q, scaler, ood, cat_categories, ranges, explainer)

arts = train_all(df)

# ----------------------------
# Inference helpers
# ----------------------------
def _prepare_df_row(d: Dict) -> pd.DataFrame:
    row = {k: d.get(k, None) for k in EXPECTED_COLUMNS if k != TARGET_COL}
    X = pd.DataFrame([row])

    # numeric
    for c in NUM_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            X[c] = X[c].fillna(df[c].median())

    # binaries (none in this setup, but kept for API symmetry)
    for c in BIN_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(int).clip(0,1)

    # cats with training categories
    for c in CAT_COLS:
        cats = arts.cat_categories[c]
        X[c] = X[c].astype(str).str.strip().str.lower()
        X[c] = pd.Categorical(X[c], categories=cats)
        if X[c].isna().any():
            if "unknown" in cats:
                X[c] = X[c].cat.add_categories(["unknown"]).fillna("unknown")
            else:
                X[c] = X[c].fillna(cats[0])
        X[c] = X[c].astype("category")

    feat_order = [c for c in EXPECTED_COLUMNS if c != TARGET_COL]
    return X[feat_order]

def predict_patient(p: Dict) -> Dict:
    X = _prepare_df_row(p)
    ps = np.stack([prob_of_one(cal, X) for cal in arts.calibrators], axis=0)
    p_mean = float(ps.mean())
    p_std = float(ps.std())
    lo = max(0.0, p_mean - arts.conformal_q)
    hi = min(1.0, p_mean + arts.conformal_q)

    # SHAP local (class 1)
    shap_vals = arts.explainer.shap_values(X)
    shap_arr = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    drivers = sorted(
        [(f, float(shap_arr[0, i])) for i, f in enumerate(X.columns)],
        key=lambda t: abs(t[1]), reverse=True
    )[:5]

    # OOD on numeric
    Xs = X[NUM_COLS].copy()
    score = arts.ood.decision_function(arts.scaler.transform(Xs))[0]  # lower -> more OOD
    ood_flag = bool(score < -0.1)

    return {
        "risk": p_mean,
        "uncertainty": {"lower": lo, "upper": hi, "epistemic_std": p_std},
        "drivers": drivers,
        "ood_flag": ood_flag
    }

def what_if(p: Dict, tweaks: Dict) -> Dict:
    after = p.copy(); after.update(tweaks)
    return {"before": predict_patient(p), "after": predict_patient(after)}

def toggle_features(p: Dict, removed: List[str]) -> Dict:
    q = p.copy()
    for f in removed:
        if f in NUM_COLS:
            q[f] = float(df[f].median())
        elif f in BIN_COLS:
            q[f] = 0
        elif f in CAT_COLS:
            cats = arts.cat_categories[f]
            q[f] = "unknown" if "unknown" in cats else cats[0]
    return predict_patient(q)

def recommend_tests(p: Dict, candidates: List[str]) -> Dict:
    base = predict_patient(p)
    base_w = base["uncertainty"]["upper"] - base["uncertainty"]["lower"]
    if base_w <= 0 or not np.isfinite(base_w):
        return {"base_width": float(base_w), "ranking": []}

    rng = np.random.default_rng(123)
    ranking = []

    for f in candidates:
        if f not in NUM_COLS:
            continue

        # start from trained ranges if available
        lo, hi = arts.ranges.get(f, (None, None))
        if lo is None or hi is None:
            lo, hi = _safe_numeric_bounds(df[f])

        # sanitize once more just before sampling
        med = float(np.nanmedian(pd.to_numeric(df[f], errors="coerce")))
        lo, hi = _sanitize_bounds(lo, hi, med)

        # sample safely
        samples = rng.uniform(float(lo), float(hi), size=60)

        widths = []
        for s in samples:
            alt = p.copy(); alt[f] = float(s)
            u = predict_patient(alt)["uncertainty"]
            width = u["upper"] - u["lower"]
            if np.isfinite(width):
                widths.append(width)

        if not widths:
            continue

        exp_w = float(np.mean(widths))
        ranking.append({
            "feature": f,
            "expected_width": exp_w,
            "expected_reduction": float(max(0.0, base_w - exp_w)),
        })

    ranking.sort(key=lambda d: d["expected_reduction"], reverse=True)
    return {"base_width": float(base_w), "ranking": ranking}

# ----------------------------
# Build a patient input
# ----------------------------
st.sidebar.header("Patient input")

input_mode = st.sidebar.radio("Input method", ["Pick a row", "Manual form"], index=0)

if input_mode == "Pick a row":
    idx = st.sidebar.number_input("Row index", min_value=0, max_value=len(df)-1, value=0, step=1)
    base_patient = df.iloc[idx].drop(TARGET_COL).to_dict()
else:
    base_patient = {}
    # categoricals
    for c in CAT_COLS:
        base_patient[c] = st.sidebar.selectbox(c, options=arts.cat_categories[c], index=0)
    # numerics
    for c in NUM_COLS:
        lo, hi = arts.ranges[c]
        base_patient[c] = st.sidebar.slider(
            f"{c} ({UNITS.get(c,'')})",
            min_value=float(lo), max_value=float(hi),
            value=float(np.mean([lo,hi]))
        )

# ----------------------------
# Prediction card
# ----------------------------
left, right = st.columns([1.1, 1.2])

with left:
    st.subheader("Prediction")
    if st.button("Run prediction", type="primary"):
        st.session_state["pred"] = predict_patient(base_patient)
    pred = st.session_state.get("pred")

    if pred:
        risk = pred["risk"]
        lo = pred["uncertainty"]["lower"]
        hi = pred["uncertainty"]["upper"]
        ep = pred["uncertainty"]["epistemic_std"]
        ood = pred["ood_flag"]

        c1,c2,c3,c4 = st.columns(4)
        to_pct = lambda x: f"{round(100*x)}%"
        c1.metric("Risk", to_pct(risk))
        c2.metric("Uncertainty band", f"{to_pct(lo)} → {to_pct(hi)}")
        c3.metric("Band width", to_pct(hi - lo))
        c4.metric("Epistemic σ", f"{ep:.3f}")

        if ood:
            st.warning("⚠️ Possible distribution shift — consider human review/alternate pathway.")

        st.markdown("**Top Drivers (local SHAP)**")
        drv = pred["drivers"]
        drv_df = pd.DataFrame(drv, columns=["feature","shap_value"]).assign(
            direction=lambda d: np.where(d["shap_value"]>=0,"↑ risk","↓ risk")
        )
        st.dataframe(drv_df, hide_index=True, use_container_width=True)

with right:
    st.subheader("What-If analysis")
    if pred:
        default_feat = "Stress_Level" if "Stress_Level" in NUM_COLS else NUM_COLS[0]
        probe_feat = st.selectbox("Feature", NUM_COLS + CAT_COLS + BIN_COLS,
                                  index=(NUM_COLS.index(default_feat) if default_feat in NUM_COLS else 0))
        if probe_feat in NUM_COLS:
            lo, hi = arts.ranges[probe_feat]
            steps = st.slider("Steps", 10, 80, 30, 5)
            xs = np.linspace(lo, hi, steps)
            risks, los, his = [], [], []
            with st.spinner("Computing curve..."):
                for x in xs:
                    res = what_if(base_patient, {probe_feat: float(x)})["after"]
                    risks.append(res["risk"])
                    los.append(res["uncertainty"]["lower"])
                    his.append(res["uncertainty"]["upper"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=his, mode="lines", name="Upper", line=dict(width=0)))
            fig.add_trace(go.Scatter(x=xs, y=los, mode="lines", name="Lower", fill='tonexty', line=dict(width=0), hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=xs, y=risks, mode="lines+markers", name="Risk"))
            fig.update_layout(
                title=f"Risk vs {probe_feat}",
                xaxis_title=f"{probe_feat} ({UNITS.get(probe_feat,'')})",
                yaxis_title="Predicted risk (0–1)",
                height=380, margin=dict(l=40,r=20,t=60,b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("For now, the what-if curve is numeric features only. Toggle categorical features below.")

st.divider()

# ----------------------------
# Feature toggle
# ----------------------------
st.subheader("Feature toggle (simulate unknown / redo)")
col_a, col_b = st.columns([1,1])
with col_a:
    removed = st.multiselect("Select features to set as 'unknown'", options=NUM_COLS+BIN_COLS+CAT_COLS, default=[])
with col_b:
    if st.button("Recompute with removed features", disabled=(len(removed)==0)):
        st.session_state["toggle"] = toggle_features(base_patient, removed)
tog = st.session_state.get("toggle")
if tog:
    to_pct = lambda x: f"{round(100*x)}%"
    c1,c2,c3 = st.columns(3)
    c1.metric("New risk", to_pct(tog["risk"]))
    c2.metric("New band", f"{to_pct(tog['uncertainty']['lower'])} → {to_pct(tog['uncertainty']['upper'])}")
    c3.metric("New width", to_pct(tog["uncertainty"]["upper"] - tog["uncertainty"]["lower"]))

st.divider()

# ----------------------------
# Recommend tests to repeat
# ----------------------------
st.subheader("Measurements to consider repeating (Expected Uncertainty Reduction)")
default_candidates = ["Stress_Level","Sleep_Hours","Work_Hours","Social_Media_Usage","Physical_Activity_Hours","Age"]
cands = st.multiselect("Candidate numeric features", options=NUM_COLS, default=[c for c in default_candidates if c in NUM_COLS])

if st.button("Rank candidates", disabled=(len(cands)==0)):
    st.session_state["reco"] = recommend_tests(base_patient, cands)

reco = st.session_state.get("reco")
if reco:
    base_w = reco["base_width"]
    df_rank = pd.DataFrame(reco["ranking"])
    if len(df_rank):
        df_rank = df_rank.rename(columns={
            "feature":"Feature",
            "expected_width":"Expected Band Width",
            "expected_reduction":"Expected Reduction"
        })
        df_rank["Reduction (%)"] = (
            (base_w - df_rank["Expected Band Width"]).clip(lower=0) / max(base_w,1e-9) * 100
        )
        st.dataframe(df_rank[["Feature","Expected Band Width","Expected Reduction","Reduction (%)"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No numeric candidates produced a calculable reduction.")
