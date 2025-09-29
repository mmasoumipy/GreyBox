import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from typing import Dict, List, Tuple
from dataclasses import dataclass

# ML bits
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import IsolationForest
import shap

st.set_page_config(page_title="Diabetes Risk + Uncertainty (Streamlit)", layout="wide")
st.title("🩺 Diabetes Risk + Uncertainty (Streamlit)")

# ----------------------------
# App constants
# ----------------------------
EXPECTED_COLUMNS = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"
]
TARGET_COL = "diabetes"
CAT_COLS = ["gender", "smoking_history"]
BIN_COLS = ["hypertension", "heart_disease"]
NUM_COLS = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]

UNITS = {
    "age": "years",
    "bmi": "kg/m²",
    "HbA1c_level": "%",
    "blood_glucose_level": "mg/dL",
    "hypertension": "0/1",
    "heart_disease": "0/1",
}

# ----------------------------
# Data loading
# ----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload diabetes_prediction_dataset.csv", type=["csv"])

@st.cache_data(show_spinner=True)
def load_default_sample() -> pd.DataFrame:
    # tiny synthetic sample if user doesn't upload (keeps app runnable)
    rng = np.random.default_rng(7)
    n = 1000
    df = pd.DataFrame({
        "gender": rng.choice(["male","female"], size=n),
        "age": rng.normal(50, 12, size=n).clip(18, 90),
        "hypertension": rng.choice([0,1], size=n, p=[0.8,0.2]),
        "heart_disease": rng.choice([0,1], size=n, p=[0.9,0.1]),
        "smoking_history": rng.choice(["never","former","current","unknown"], size=n, p=[0.5,0.2,0.2,0.1]),
        "bmi": rng.normal(27, 5, size=n).clip(15, 55),
        "HbA1c_level": rng.normal(5.8, 1.0, size=n).clip(4.5, 12.5),
        "blood_glucose_level": rng.normal(115, 30, size=n).clip(60, 260),
    })
    # synthetic label
    logit = (
        -7.0
        + 0.04*df["age"]
        + 0.08*(df["bmi"]-25)
        + 0.7*(df["hypertension"])
        + 1.2*np.maximum(df["HbA1c_level"]-6.5, 0)
        + 0.006*(df["blood_glucose_level"]-100)
    )
    p = 1 / (1 + np.exp(-logit))
    df["diabetes"] = (rng.uniform(0, 1, size=n) < p).astype(int)
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
df["gender"] = df["gender"].astype(str).str.strip().str.lower()
df["smoking_history"] = df["smoking_history"].astype(str).str.strip().str.lower()
for c in BIN_COLS + [TARGET_COL]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

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

    p_val = np.mean([cal.predict_proba(X_val)[:,1] for cal in calibrators], axis=0)
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

    # numeric slider ranges (2nd–98th)
    ranges = {}
    for c in NUM_COLS:
        lo, hi = np.percentile(X_tr[c], [2,98])
        if hi <= lo: hi = lo + 1.0
        ranges[c] = [float(lo), float(hi)]

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

    # binaries
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

    # order
    feat_order = [c for c in EXPECTED_COLUMNS if c != TARGET_COL]
    return X[feat_order]

def predict_patient(p: Dict) -> Dict:
    X = _prepare_df_row(p)
    ps = np.stack([cal.predict_proba(X)[:,1] for cal in arts.calibrators], axis=0)
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
    if base_w <= 0:
        return {"base_width": base_w, "ranking": []}
    rng = np.random.default_rng(123)
    ranking = []
    for f in candidates:
        if f not in NUM_COLS:  # simple MVP on numeric
            continue
        lo, hi = arts.ranges.get(f, [float(df[f].quantile(0.02)), float(df[f].quantile(0.98))])
        samples = rng.uniform(lo, hi, size=60)
        widths = []
        for s in samples:
            alt = p.copy(); alt[f] = float(s)
            u = predict_patient(alt)["uncertainty"]
            widths.append(u["upper"] - u["lower"])
        exp_w = float(np.mean(widths))
        ranking.append({
            "feature": f,
            "expected_width": exp_w,
            "expected_reduction": float(max(0.0, base_w - exp_w))
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
    # binaries
    for c in BIN_COLS:
        base_patient[c] = st.sidebar.selectbox(c, options=[0,1], index=0)
    # numerics
    for c in NUM_COLS:
        lo, hi = arts.ranges[c]
        base_patient[c] = st.sidebar.slider(f"{c} ({UNITS.get(c,'')})", min_value=float(lo), max_value=float(hi),
                                            value=float(np.mean([lo,hi])))

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
        probe_feat = st.selectbox("Feature", NUM_COLS + CAT_COLS + BIN_COLS, index=NUM_COLS.index("HbA1c_level") if "HbA1c_level" in NUM_COLS else 0)
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
            st.info("For MVP, the what-if curve is numeric features only. Toggle binary/categorical below.")

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
st.subheader("Tests to consider repeating (Expected Uncertainty Reduction)")
default_candidates = ["HbA1c_level","blood_glucose_level","bmi","age"]
cands = st.multiselect("Candidate numeric features", options=NUM_COLS, default=[c for c in default_candidates if c in NUM_COLS])

if st.button("Rank candidates", disabled=(len(cands)==0)):
    st.session_state["reco"] = recommend_tests(base_patient, cands)

reco = st.session_state.get("reco")
if reco:
    base_w = reco["base_width"]
    df_rank = pd.DataFrame(reco["ranking"])
    if len(df_rank):
        df_rank["Reduction (%)"] = (base_w - df_rank["expected_width"]).clip(lower=0) / max(base_w,1e-9) * 100
        df_rank = df_rank.rename(columns={"feature":"Feature","expected_width":"Expected Band Width","expected_reduction":"Expected Reduction"})
        st.dataframe(df_rank[["Feature","Expected Band Width","Expected Reduction","Reduction (%)"]], use_container_width=True, hide_index=True)
    else:
        st.info("No numeric candidates produced a calculable reduction.")
