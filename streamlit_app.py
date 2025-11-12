import ctypes.util
import sys
from typing import Dict, List
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

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

st.set_page_config(page_title="Stress Risk Assessment", layout="wide")

# ----------------------------
# Session state initialization
# ----------------------------
if "study_mode" not in st.session_state:
    st.session_state["study_mode"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = ""
if "interaction_log" not in st.session_state:
    st.session_state["interaction_log"] = []

# ----------------------------
# Study mode selection
# ----------------------------
st.title("🧠 Stress Risk Assessment System")

if st.session_state["study_mode"] is None:
    st.markdown("""
    ### Welcome to the Stress Risk Assessment Study
    
    This system helps assess stress risk based on lifestyle, work, and health metrics.
    
    **Please select your study group:**
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔵 Group 1 (Basic Assessment)", use_container_width=True):
            st.session_state["study_mode"] = "G1"
            st.rerun()
    
    with col2:
        if st.button("🟢 Group 2 (Enhanced Assessment)", use_container_width=True):
            st.session_state["study_mode"] = "G2"
            st.rerun()
    
    st.info("💡 Your group assignment will determine which features you'll see during the assessment.")
    st.stop()

# User ID input
if not st.session_state["user_id"]:
    st.markdown("### Participant Information")
    user_id = st.text_input("Enter your participant ID:", key="uid_input")
    if st.button("Continue"):
        if user_id.strip():
            st.session_state["user_id"] = user_id.strip()
            st.session_state["interaction_log"].append({
                "timestamp": pd.Timestamp.now().isoformat(),
                "event": "session_start",
                "user_id": user_id.strip(),
                "group": st.session_state["study_mode"]
            })
            st.rerun()
        else:
            st.warning("Please enter a valid participant ID")
    st.stop()

# Display current mode
mode_label = "Basic Assessment" if st.session_state["study_mode"] == "G1" else "Enhanced Assessment"
st.sidebar.success(f"Mode: {mode_label}")
st.sidebar.caption(f"Participant: {st.session_state['user_id']}")

# ----------------------------
# Safety helpers
# ----------------------------
def prob_of_one(cal: CalibratedClassifierCV, X: pd.DataFrame) -> np.ndarray:
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
# Stress management plan generator
# ----------------------------
def generate_stress_management_plan(user_data: Dict, prediction: Dict) -> Dict:
    """Generate personalized stress management plan based on user metrics."""
    
    # Extract key metrics
    stress_events = user_data.get("stress_event_count_last_week", 2)
    job_satisfaction = user_data.get("job_satisfaction", 5)
    workload_rating = user_data.get("workload_rating", 5)
    sleep_quality = user_data.get("sleep_quality_rating", 5)
    sleep_duration = user_data.get("sleep_duration_hr", 7)
    exercise_frequency = user_data.get("physical_activity_frequency", 3)
    coffee_intake = user_data.get("coffee_intake_cups", 2)
    alcohol_intake = user_data.get("alcohol_intake_per_week", 2)
    screen_time = user_data.get("screen_time_hr", 5)
    social_interactions = user_data.get("social_interactions_count", 5)
    breaks_per_day = user_data.get("breaks_per_workday", 2)
    work_hours = user_data.get("work_hours_per_week", 40)
    
    # Identify stress factors
    stress_factors = []
    
    if stress_events >= 3:
        stress_factors.append("high_stress_events")
    if job_satisfaction < 4:
        stress_factors.append("low_job_satisfaction")
    if workload_rating >= 7:
        stress_factors.append("high_workload")
    if sleep_quality < 5:
        stress_factors.append("poor_sleep_quality")
    if sleep_duration < 7:
        stress_factors.append("insufficient_sleep")
    if exercise_frequency < 3:
        stress_factors.append("low_exercise")
    if coffee_intake > 3:
        stress_factors.append("high_caffeine")
    if alcohol_intake > 7:
        stress_factors.append("high_alcohol")
    if screen_time > 6:
        stress_factors.append("excessive_screen_time")
    if social_interactions < 3:
        stress_factors.append("low_social_contact")
    if breaks_per_day < 3:
        stress_factors.append("insufficient_breaks")
    if work_hours > 50:
        stress_factors.append("overwork")
    
    # If no major factors, focus on maintenance
    if not stress_factors:
        stress_factors = ["maintenance", "prevention"]
    
    plan = {
        "risk_factors": stress_factors,
        "daily_routine": {}
    }
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    for i, day in enumerate(days):
        activities = []
        
        # Morning routine
        activities.append({
            "time": "6:30-7:00 AM",
            "activity": "Wake up routine - hydrate, light stretching or yoga",
            "category": "Morning",
            "duration": "30 min"
        })
        
        # Work preparation
        activities.append({
            "time": "7:00-8:00 AM",
            "activity": "Healthy breakfast - avoid excessive coffee, plan priorities",
            "category": "Breakfast",
            "duration": "60 min"
        })
        
        # Workday stress management
        if "insufficient_breaks" in stress_factors or "high_workload" in stress_factors:
            activities.append({
                "time": "9:30 AM",
                "activity": "First break - 5 min breathing exercise or walk",
                "category": "Break",
                "duration": "5 min"
            })
            activities.append({
                "time": "12:00 PM",
                "activity": "Lunch break - step outside or change scenery",
                "category": "Lunch",
                "duration": "30 min"
            })
            activities.append({
                "time": "3:00 PM",
                "activity": "Afternoon break - hydration + 5 min movement",
                "category": "Break",
                "duration": "5 min"
            })
        
        # Exercise
        if "low_exercise" in stress_factors:
            if i < 5:  # Weekdays
                activities.append({
                    "time": "5:30 PM" if i % 2 == 0 else "6:30 AM (before work)",
                    "activity": "Moderate exercise - 30-45 min (walk, gym, cycling, yoga)",
                    "category": "Exercise",
                    "duration": "45 min"
                })
            else:  # Weekends
                activities.append({
                    "time": "9:00 AM",
                    "activity": "Longer activity - 60 min outdoor activity or sports",
                    "category": "Exercise",
                    "duration": "60 min"
                })
        
        # Social connection
        if "low_social_contact" in stress_factors:
            if i in [2, 5]:
                activities.append({
                    "time": "7:00 PM",
                    "activity": f"Social time - {'meet friend for coffee' if i == 2 else 'family dinner or video call'}",
                    "category": "Social",
                    "duration": "60 min"
                })
        
        # Caffeine management
        if "high_caffeine" in stress_factors:
            activities.append({
                "time": "Before 2 PM",
                "activity": "Limit coffee to 1-2 cups in morning (cut off by 2 PM)",
                "category": "Nutrition",
                "duration": "Ongoing"
            })
        
        # Screen time management
        if "excessive_screen_time" in stress_factors:
            activities.append({
                "time": "8:30-9:00 PM",
                "activity": "Digital sunset - put phone away, reduce screen exposure",
                "category": "Digital Wellness",
                "duration": "30 min"
            })
        
        # Evening routine
        if "poor_sleep_quality" in stress_factors or "insufficient_sleep" in stress_factors:
            activities.append({
                "time": "9:00-10:00 PM",
                "activity": "Wind-down routine - read, light stretching, meditation",
                "category": "Evening",
                "duration": "60 min"
            })
            activities.append({
                "time": "10:00 PM",
                "activity": "Consistent bedtime - aim for 7-8 hours sleep",
                "category": "Sleep",
                "duration": "N/A"
            })
        
        plan["daily_routine"][day] = activities
    
    # Weekly goals
    plan["weekly_goals"] = []
    
    if "high_stress_events" in stress_factors:
        plan["weekly_goals"].append({
            "goal": "Practice stress management techniques",
            "target": "Daily",
            "tips": ["Meditation", "Deep breathing", "Journaling"]
        })
    
    if "insufficient_sleep" in stress_factors:
        plan["weekly_goals"].append({
            "goal": "Achieve 7-8 hours sleep",
            "target": "5+ nights",
            "tips": ["Consistent bedtime", "Dark room", "No screens 1 hour before bed"]
        })
    
    if "low_exercise" in stress_factors:
        plan["weekly_goals"].append({
            "goal": "Physical activity",
            "target": "150 min moderate activity",
            "tips": ["Brisk walking", "Yoga", "Cycling", "Team sports"]
        })
    
    if "high_caffeine" in stress_factors:
        plan["weekly_goals"].append({
            "goal": "Reduce caffeine",
            "target": "Max 1-2 cups before noon",
            "tips": ["Switch to herbal tea", "Gradual reduction", "Track intake"]
        })
    
    if "excessive_screen_time" in stress_factors:
        plan["weekly_goals"].append({
            "goal": "Digital wellness",
            "target": "Reduce by 2 hours/day",
            "tips": ["App time trackers", "Digital sunset", "More outdoor time"]
        })
    
    if "low_social_contact" in stress_factors:
        plan["weekly_goals"].append({
            "goal": "Social connection",
            "target": "3+ meaningful interactions",
            "tips": ["Phone calls", "In-person meetups", "Group activities"]
        })
    
    return plan

def display_stress_plan(plan: Dict):
    """Display the stress management plan."""
    
    st.markdown("### 🎯 Your Stress Risk Factors")
    
    factor_map = {
        "high_stress_events": "⚠️ High Stress Events",
        "low_job_satisfaction": "😞 Low Job Satisfaction",
        "high_workload": "📊 High Workload",
        "poor_sleep_quality": "😴 Poor Sleep Quality",
        "insufficient_sleep": "⏰ Insufficient Sleep",
        "low_exercise": "🏃 Low Exercise",
        "high_caffeine": "☕ High Caffeine",
        "high_alcohol": "🍷 High Alcohol",
        "excessive_screen_time": "📱 Excessive Screen Time",
        "low_social_contact": "👥 Low Social Contact",
        "insufficient_breaks": "🚫 Insufficient Breaks",
        "overwork": "💼 Overwork",
        "maintenance": "✨ Wellness Maintenance",
        "prevention": "🛡️ Stress Prevention"
    }
    
    cols = st.columns(min(len(plan["risk_factors"]), 4))
    for i, factor in enumerate(plan["risk_factors"]):
        with cols[i % 4]:
            st.warning(factor_map.get(factor, factor))
    
    st.markdown("### 📊 Weekly Goals")
    for goal in plan["weekly_goals"]:
        with st.expander(f"🎯 {goal['goal']}"):
            st.markdown(f"**Target:** {goal['target']}")
            st.markdown("**Tips:**")
            for tip in goal['tips']:
                st.markdown(f"- {tip}")
    
    st.markdown("### 📅 Daily Action Plan")
    
    days = list(plan["daily_routine"].keys())
    tabs = st.tabs(days)
    
    for tab, day in zip(tabs, days):
        with tab:
            activities = plan["daily_routine"][day]
            for activity in activities:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**{activity['time']}**")
                with col2:
                    st.markdown(f"{activity['activity']}")
                    st.caption(f"⏱️ {activity['duration']}")
            st.markdown("")
    
    # Tips section
    st.markdown("### 💡 Implementation Tips")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Start Small:**
        - Pick 1-2 changes to start
        - Build gradually
        - Track progress
        """)
    
    with col2:
        st.markdown("""
        **Stay Consistent:**
        - Set reminders
        - Find accountability partner
        - Celebrate wins
        """)
    
    # Download
    st.markdown("### 📥 Download Your Plan")
    plan_text = "PERSONALIZED STRESS MANAGEMENT PLAN\n\n"
    plan_text += "=" * 50 + "\n"
    plan_text += "RISK FACTORS:\n"
    for factor in plan["risk_factors"]:
        plan_text += f"- {factor_map.get(factor, factor)}\n"
    plan_text += "\n"
    
    for day, activities in plan["daily_routine"].items():
        plan_text += f"\n{day.upper()}\n"
        plan_text += "-" * 30 + "\n"
        for act in activities:
            plan_text += f"{act['time']}: {act['activity']} ({act['duration']})\n"
    
    st.download_button(
        label="📄 Download Plan as Text",
        data=plan_text,
        file_name=f"stress_plan_{st.session_state['user_id']}.txt",
        mime="text/plain",
        use_container_width=True
    )

def save_interaction_log():
    """Save interaction log to JSON file."""
    log_dir = Path("study_logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{st.session_state['user_id']}_{st.session_state['study_mode']}_{timestamp}.json"
    
    with open(log_dir / filename, 'w') as f:
        json.dump(st.session_state["interaction_log"], f, indent=2)
    
    st.sidebar.success(f"✅ Session data saved")

# ----------------------------
# Data loading
# ----------------------------
st.sidebar.header("📊 Data Management")
uploaded = st.sidebar.file_uploader("Upload stress risk dataset (CSV)", type=["csv"])

@st.cache_data(show_spinner=True)
def load_default_sample() -> pd.DataFrame | None:
    """Load default Stress Risk Dataset."""
    try:
        df = pd.read_csv("data/stress_risk_data.csv")
        return df
    except Exception:
        st.error("Could not load default dataset. Please upload a CSV file.")
        return None

if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("✅ Dataset loaded from upload.")
else:
    df = load_default_sample()
    if df is not None:
        st.sidebar.info("Using stress risk dataset.")

if df is None:
    st.stop()

# Define feature columns
NUM_COLS = [c for c in df.columns if c not in ["user_id", "stress_risk_score", 
                                               "stress_level_category", "data_split", "gender", "occupation"]]
CAT_COLS = ["gender", "occupation"]
TARGET_COL = "stress_risk_score"

# Clean data
df = df[[c for c in df.columns if c not in ["user_id", "data_split"]]].copy()

# ---- categorical cleaning + casting (CRITICAL) ----
for c in CAT_COLS:
    if c in df.columns:
        df[c] = (
            df[c]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": np.nan})
        )
        df[c] = df[c].astype("category")  # make it a pandas categorical

# numeric cleaning
for c in NUM_COLS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

if TARGET_COL in df.columns:
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df["target_binary"] = (df[TARGET_COL] > df[TARGET_COL].median()).astype(int)

# ----------------------------
# Model training
# ----------------------------
@dataclass
class Artifacts:
    models: List[lgb.LGBMClassifier]
    calibrators: List[CalibratedClassifierCV]
    conformal_q: float
    conformal_eps: float
    scaler: StandardScaler
    ood: IsolationForest
    cat_categories: Dict[str, List[str]]
    ranges: Dict[str, List[float]]
    explainer: shap.TreeExplainer
    feature_cols: List[str]

@st.cache_resource(show_spinner=True)
def train_all(df: pd.DataFrame) -> Artifacts:
    # ensure categorical dtype present before splitting
    feature_cols = [c for c in NUM_COLS + CAT_COLS if c in df.columns]
    X = df[feature_cols].copy()

    # enforce categorical dtype for declared categorical columns
    for c in [col for col in CAT_COLS if col in X.columns]:
        if not pd.api.types.is_categorical_dtype(X[c]):
            X[c] = X[c].astype("category")

    y = df["target_binary"].astype(int)

    # split (stratify if possible)
    if y.nunique() < 2 or y.value_counts().min() < 5:
        X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
        X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    else:
        X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=42)
        X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    # keep categorical dtype after split
    for frame in (X_tr, X_val, X_te):
        for c in [col for col in CAT_COLS if col in frame.columns]:
            if not pd.api.types.is_categorical_dtype(frame[c]):
                frame[c] = frame[c].astype("category")

    # collect training categories and add an 'unknown' bucket for safety
    cat_categories: Dict[str, List[str]] = {}
    for c in CAT_COLS:
        if c in X_tr.columns:
            cats = list(pd.Categorical(X_tr[c]).categories)
            if "unknown" not in cats:
                cats = cats + ["unknown"]
            # re-cast with explicit categories to lock them in
            X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
            X_val[c] = pd.Categorical(X_val[c], categories=cats)
            X_te[c]  = pd.Categorical(X_te[c],  categories=cats)
            cat_categories[c] = cats

    def fit_lgbm(Xdf, y, seed):
        cat_feats = [c for c in CAT_COLS if c in Xdf.columns]
        clf = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1
        )
        clf.fit(
            Xdf, y,
            categorical_feature=cat_feats,   # names are OK with pandas DataFrame
            eval_set=[(Xdf, y)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        return clf

    models = [fit_lgbm(X_tr, y_tr, s) for s in [11, 22, 33, 44, 55]]

    calibrators: List[CalibratedClassifierCV] = []
    p_val_stack = []
    for m in models:
        cal = CalibratedClassifierCV(m, method="isotonic", cv="prefit")
        cal.fit(X_val, y_val)
        calibrators.append(cal)
        p_val_stack.append(prob_of_one(cal, X_val))

    p_val_stack = np.stack(p_val_stack, axis=0)
    p_val = np.mean(p_val_stack, axis=0)
    ens_std_val = np.std(p_val_stack, axis=0)
    aleatoric_val = np.mean(p_val_stack * (1 - p_val_stack), axis=0)
    
    alpha = 0.10
    eps = 1e-6
    total_scale = np.sqrt(np.clip(aleatoric_val + ens_std_val**2, 0.0, None) + eps)
    val_scores = np.abs(y_val.values - p_val) / total_scale
    q = float(np.quantile(val_scores, 1 - alpha))

    # OOD on numeric features
    num_feats_present = [c for c in NUM_COLS if c in X_tr.columns]
    if len(num_feats_present) == 0:
        # ensure scaler/ood still exist even if no numeric columns (unlikely)
        scaler = StandardScaler().fit(np.zeros((len(X_tr), 1)))
        Xtr_scaled = scaler.transform(np.zeros((len(X_tr), 1)))
    else:
        scaler = StandardScaler().fit(X_tr[num_feats_present])
        Xtr_scaled = scaler.transform(X_tr[num_feats_present])

    ood = IsolationForest(n_estimators=300, contamination=0.05, random_state=7)
    ood.fit(Xtr_scaled)

    # numeric ranges for UI what-if (optional)
    ranges = {}
    for c in NUM_COLS:
        if c in X_tr.columns:
            vals = X_tr[c].dropna()
            if len(vals) > 0:
                lo, hi = np.percentile(vals, [2, 98])
                if hi <= lo:
                    hi = lo + 1.0
                ranges[c] = [float(lo), float(hi)]

    # SHAP explainer
    explainer = shap.TreeExplainer(models[0])
    
    return Artifacts(models, calibrators, q, eps, scaler, ood, cat_categories, ranges, explainer, feature_cols)

with st.spinner("🔄 Training model..."):
    arts = train_all(df)

st.sidebar.success("✅ Model trained successfully")

# ----------------------------
# Inference
# ----------------------------
def _prepare_df_row(d: Dict) -> pd.DataFrame:
    row = {k: d.get(k, None) for k in arts.feature_cols}
    X = pd.DataFrame([row])

    # numeric fill
    for c in [col for col in NUM_COLS if col in X.columns]:
        # fallback to median from training df if present, else 0
        if c in df.columns:
            med = df[c].median()
        else:
            med = 0.0
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med)

    # categorical: normalize + enforce training categories with 'unknown'
    for c in [col for col in CAT_COLS if col in X.columns]:
        X[c] = X[c].astype(str).str.strip().str.lower()
        cats = arts.cat_categories.get(c, [])
        if cats:
            X[c] = pd.Categorical(X[c], categories=cats)
            if X[c].isna().any():
                X[c] = X[c].fillna("unknown")
        else:
            X[c] = pd.Categorical(X[c])

    # keep column order
    return X[arts.feature_cols]

def predict_user(p: Dict) -> Dict:
    X = _prepare_df_row(p)
    ps = np.stack([prob_of_one(cal, X) for cal in arts.calibrators], axis=0)
    p_mean = float(ps.mean())
    p_std = float(ps.std())
    aleatoric = float(np.mean(ps * (1 - ps)))
    total_scale = float(np.sqrt(max(aleatoric + p_std**2, 0.0) + arts.conformal_eps))
    delta = arts.conformal_q * total_scale
    lo = max(0.0, p_mean - delta)
    hi = min(1.0, p_mean + delta)

    shap_vals = arts.explainer.shap_values(X)
    shap_arr = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    drivers = sorted(
        [(f, float(shap_arr[0, i])) for i, f in enumerate(X.columns)],
        key=lambda t: abs(t[1]), reverse=True
    )[:5]

    # OOD on numeric columns only
    num_feats_present = [c for c in arts.feature_cols if c in NUM_COLS and c in X.columns]
    if len(num_feats_present) == 0:
        Xs = np.zeros((1, 1))
    else:
        Xs = arts.scaler.transform(X[num_feats_present])
    score = arts.ood.decision_function(Xs)[0]
    ood_flag = bool(score < -0.1)

    return {
        "risk": p_mean,
        "uncertainty": {
            "lower": lo,
            "upper": hi,
            "aleatoric_std": float(np.sqrt(max(aleatoric, 0.0))),
            "epistemic_std": p_std,
        },
        "drivers": drivers,
        "ood_flag": ood_flag
    }

# ----------------------------
# User input form
# ----------------------------
st.markdown("---")
st.header("📝 Your Information")

with st.form("user_form"):
    st.subheader("Demographics & Work")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = st.number_input("Age", 18, 80, 30)
    with col2:
        gender_choices = arts.cat_categories.get("gender", ["male", "female", "other"])
        gender = st.selectbox("Gender", gender_choices)
    with col3:
        occ_choices = arts.cat_categories.get("occupation", ["engineer", "nurse", "student"])
        occupation = st.selectbox("Occupation", occ_choices)
    with col4:
        work_hours = st.number_input("Work Hours/Week", 0, 80, 40)
    
    st.subheader("Work & Stress Factors")
    col5, col6, col7 = st.columns(3)
    
    with col5:
        job_satisfaction = st.slider("Job Satisfaction (1-10)", 1, 10, 5)
        workload = st.slider("Workload Rating (1-10)", 1, 10, 5)
    with col6:
        stress_events = st.slider("Stress Events (Last Week)", 0, 10, 2)
        breaks = st.slider("Breaks/Workday", 0, 6, 3)
    with col7:
        commute = st.number_input("Commute Time (min)", 0, 180, 30)
        outdoor_time = st.slider("Outdoor Time (hr/day)", 0.0, 8.0, 1.0, 0.5)
    
    st.subheader("Sleep & Health")
    col8, col9 = st.columns(2)
    
    with col8:
        sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 7)
        sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 12.0, 7.0, 0.5)
        exercise = st.slider("Physical Activity (times/week)", 0, 7, 3)
    
    with col9:
        coffee = st.number_input("Coffee (cups/day)", 0.0, 10.0, 2.0)
        alcohol = st.number_input("Alcohol (drinks/week)", 0, 20, 2)
        screen_time = st.slider("Screen Time (hr/day)", 0.0, 16.0, 5.0, 0.5)
    
    st.subheader("Lifestyle")
    col10, col11 = st.columns(2)
    
    with col10:
        social = st.slider("Social Interactions (count/week)", 0, 30, 8)
        screen_unlocks = st.number_input("Screen Unlocks/Day", 0, 500, 80)
    
    submitted = st.form_submit_button("🔍 Assess Stress Risk", type="primary", use_container_width=True)

if submitted:
    user_data = {
        "age": age,
        "gender": gender,
        "occupation": occupation,
        "work_hours_per_week": work_hours,
        "job_satisfaction": job_satisfaction,
        "workload_rating": workload,
        "stress_event_count_last_week": stress_events,
        "breaks_per_workday": breaks,
        "commute_time_min": commute,
        "outdoor_time_hr": outdoor_time,
        "sleep_quality_rating": sleep_quality,
        "sleep_duration_hr": sleep_duration,
        "physical_activity_frequency": exercise,
        "coffee_intake_cups": coffee,
        "alcohol_intake_per_week": alcohol,
        "screen_time_hr": screen_time,
        "social_interactions_count": social,
        "screen_unlocks_per_day": screen_unlocks
    }
    
    pred = predict_user(user_data)
    st.session_state["current_prediction"] = pred
    st.session_state["current_user"] = user_data
    
    st.session_state["interaction_log"].append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "event": "prediction_requested",
        "user_id": st.session_state["user_id"],
        "group": st.session_state["study_mode"],
        "user_data": user_data,
        "prediction": pred
    })

# ----------------------------
# Results display
# ----------------------------
if "current_prediction" in st.session_state:
    st.markdown("---")
    st.header("📊 Stress Risk Assessment Results")
    
    pred = st.session_state["current_prediction"]
    risk = pred["risk"]
    
    # G1: Basic display
    if st.session_state["study_mode"] == "G1":
        st.markdown("### Your Stress Risk Score")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk * 100,
            title = {'text': "Stress Risk Level (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "salmon"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        if risk < 0.33:
            st.success("✅ Low Stress Risk")
        elif risk < 0.66:
            st.warning("⚠️ Moderate Stress Risk")
        else:
            st.error("🚨 High Stress Risk")
    
    # G2: Enhanced display
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Stress Risk with Uncertainty")
            
            lo = pred["uncertainty"]["lower"]
            hi = pred["uncertainty"]["upper"]
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk * 100,
                title = {'text': "Stress Risk Level (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgreen"},
                        {'range': [33, 66], 'color': "yellow"},
                        {'range': [66, 100], 'color': "salmon"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Uncertainty Metrics")
            unc_width = hi - lo
            c1, c2, c3 = st.columns(3)
            c1.metric("Lower Bound", f"{lo*100:.1f}%")
            c2.metric("Upper Bound", f"{hi*100:.1f}%")
            c3.metric("Range", f"±{unc_width*50:.1f}%")
        
        with col2:
            st.markdown("### Key Stress Drivers")
            st.markdown("Factors with the most impact on your risk score:")
            
            drivers = pred["drivers"]
            driver_df = pd.DataFrame(drivers, columns=["Factor", "Impact"])
            driver_df["Direction"] = driver_df["Impact"].apply(
                lambda x: "Increases ↑" if x > 0 else "Decreases ↓"
            )
            
            fig_shap = px.bar(
                driver_df, y="Factor", x="Impact", orientation='h',
                color="Direction",
                color_discrete_map={"Increases ↑": "#ff6b6b", "Decreases ↓": "#51cf66"}
            )
            fig_shap.update_layout(height=350, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_shap, use_container_width=True)
    
    # Stress management plan
    st.markdown("---")
    st.header("📅 Your Personalized Stress Management Plan")
    
    if st.button("Generate My Plan", type="primary", use_container_width=True):
        plan = generate_stress_management_plan(st.session_state["current_user"], pred)
        st.session_state["stress_plan"] = plan
        
        st.session_state["interaction_log"].append({
            "timestamp": pd.Timestamp.now().isoformat(),
            "event": "plan_generated",
            "user_id": st.session_state["user_id"],
            "group": st.session_state["study_mode"]
        })
    
    if "stress_plan" in st.session_state:
        display_stress_plan(st.session_state["stress_plan"])
        
        # Feedback survey
        st.markdown("---")
        st.header("📋 Feedback Survey")
        
        with st.form("feedback_form"):
            trust = st.slider("Trust in this assessment?", 1, 5, 3)
            follow = st.slider("Likelihood to follow recommendations?", 1, 5, 3)
            useful = st.slider("Usefulness of the assessment?", 1, 5, 3)
            comments = st.text_area("Additional comments?", placeholder="Your feedback...")
            
            survey_submit = st.form_submit_button("Submit", use_container_width=True)
        
        if survey_submit:
            st.session_state["interaction_log"].append({
                "timestamp": pd.Timestamp.now().isoformat(),
                "event": "survey_completed",
                "user_id": st.session_state["user_id"],
                "group": st.session_state["study_mode"],
                "trust_score": trust,
                "follow_likelihood": follow,
                "usefulness": useful,
                "comments": comments
            })
            
            st.success("✅ Thank you for your feedback!")
            save_interaction_log()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>This is a research tool for educational purposes.</p>
    <p>Your data is anonymized and used only for research.</p>
</div>
""", unsafe_allow_html=True)
