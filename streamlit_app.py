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

st.set_page_config(page_title="Mental Health Risk Assessment", layout="wide")

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
st.title("🧠 Mental Health Risk Assessment System")

if st.session_state["study_mode"] is None:
    st.markdown("""
    ### Welcome to the Mental Health Risk Assessment Study
    
    This system helps assess mental health risk based on lifestyle and demographic factors.
    
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
# Wellness plan generator
# ----------------------------
def generate_wellness_plan(patient_data: Dict, prediction: Dict) -> Dict:
    """Generate personalized 7-day wellness plan based on patient data and risk factors."""
    
    # Extract key metrics
    stress = patient_data.get("Stress_Level", 5)
    sleep = patient_data.get("Sleep_Hours", 7)
    exercise = patient_data.get("Physical_Activity_Hours", 3)
    social_media = patient_data.get("Social_Media_Usage", 2)
    diet = patient_data.get("Diet_Quality", "average")
    work_hours = patient_data.get("Work_Hours", 40)
    
    # Identify top areas for improvement
    improvements = []
    
    if stress >= 7:
        improvements.append("stress_management")
    if sleep < 7:
        improvements.append("sleep_hygiene")
    if exercise < 3:
        improvements.append("physical_activity")
    if social_media > 4:
        improvements.append("digital_wellness")
    if diet in ["unhealthy", "poor"]:
        improvements.append("nutrition")
    if work_hours > 50:
        improvements.append("work_life_balance")
    
    # If no major issues, focus on maintenance
    if not improvements:
        improvements = ["maintenance", "preventive_care"]
    
    # Generate daily plan
    plan = {
        "focus_areas": improvements,
        "daily_schedule": {}
    }
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    for i, day in enumerate(days):
        daily_activities = []
        
        # Morning routine
        if "sleep_hygiene" in improvements:
            daily_activities.append({
                "time": "7:00 AM",
                "activity": "Wake up at consistent time",
                "duration": "N/A",
                "category": "Sleep"
            })
        
        daily_activities.append({
            "time": "7:30 AM",
            "activity": "Morning sunlight exposure (10 min walk or sit by window)",
            "duration": "10 min",
            "category": "Circadian"
        })
        
        # Stress management
        if "stress_management" in improvements:
            if i % 2 == 0:
                daily_activities.append({
                    "time": "8:00 AM",
                    "activity": "Guided meditation or deep breathing exercises",
                    "duration": "10-15 min",
                    "category": "Mental Health"
                })
            else:
                daily_activities.append({
                    "time": "12:00 PM",
                    "activity": "Mindfulness break - body scan or gratitude journaling",
                    "duration": "10 min",
                    "category": "Mental Health"
                })
        
        # Nutrition
        if "nutrition" in improvements:
            daily_activities.append({
                "time": "Meal times",
                "activity": "Include vegetables in at least 2 meals, stay hydrated (8 glasses water)",
                "duration": "Throughout day",
                "category": "Nutrition"
            })
        
        # Physical activity
        if "physical_activity" in improvements:
            if i < 5:  # Weekdays
                daily_activities.append({
                    "time": "6:00 PM",
                    "activity": f"{'Moderate exercise (brisk walk, yoga, cycling)' if i % 2 == 0 else 'Light stretching or gentle movement'}",
                    "duration": "30 min",
                    "category": "Physical"
                })
            else:  # Weekends
                daily_activities.append({
                    "time": "10:00 AM",
                    "activity": "Longer outdoor activity (hiking, sports, nature walk)",
                    "duration": "60 min",
                    "category": "Physical"
                })
        
        # Digital wellness
        if "digital_wellness" in improvements:
            daily_activities.append({
                "time": "9:00 PM",
                "activity": "Begin digital sunset - reduce screen time, no social media",
                "duration": "Until bedtime",
                "category": "Digital Wellness"
            })
        
        # Work-life balance
        if "work_life_balance" in improvements and i < 5:
            daily_activities.append({
                "time": "5:00 PM",
                "activity": "Hard stop work - set boundaries, transition to personal time",
                "duration": "N/A",
                "category": "Work-Life Balance"
            })
        
        # Social connection
        if i in [2, 5]:
            daily_activities.append({
                "time": "7:00 PM" if i == 2 else "2:00 PM",
                "activity": "Social connection - call friend/family or meet in person",
                "duration": "30-60 min",
                "category": "Social"
            })
        
        # Evening routine
        if "sleep_hygiene" in improvements:
            daily_activities.append({
                "time": "10:00 PM",
                "activity": "Wind-down routine: dim lights, read, or light stretching",
                "duration": "30 min",
                "category": "Sleep"
            })
            daily_activities.append({
                "time": "10:30 PM",
                "activity": "Bedtime - aim for 7-8 hours sleep",
                "duration": "N/A",
                "category": "Sleep"
            })
        
        plan["daily_schedule"][day] = daily_activities
    
    # Weekly goals
    plan["weekly_goals"] = []
    
    if "stress_management" in improvements:
        plan["weekly_goals"].append({
            "goal": "Practice stress management techniques daily",
            "target": "7 days",
            "metric": "sessions"
        })
    
    if "sleep_hygiene" in improvements:
        plan["weekly_goals"].append({
            "goal": "Achieve 7-8 hours of sleep",
            "target": "5+ nights",
            "metric": "nights"
        })
    
    if "physical_activity" in improvements:
        plan["weekly_goals"].append({
            "goal": "Complete at least 150 minutes of moderate activity",
            "target": "150 min",
            "metric": "minutes"
        })
    
    if "digital_wellness" in improvements:
        plan["weekly_goals"].append({
            "goal": "Limit social media to under 2 hours daily",
            "target": "< 2 hrs/day",
            "metric": "hours"
        })
    
    if "nutrition" in improvements:
        plan["weekly_goals"].append({
            "goal": "Eat balanced meals with vegetables and fruits",
            "target": "2-3 servings/day",
            "metric": "servings"
        })
    
    if "work_life_balance" in improvements:
        plan["weekly_goals"].append({
            "goal": "Limit work hours to 50 per week",
            "target": "≤ 50 hrs",
            "metric": "hours"
        })
    
    return plan

def display_wellness_plan(plan: Dict):
    """Display the wellness plan in an organized, user-friendly format."""
    
    st.markdown("### 🎯 Your Focus Areas This Week")
    
    focus_map = {
        "stress_management": "🧘 Stress Management",
        "sleep_hygiene": "😴 Sleep Quality",
        "physical_activity": "🏃 Physical Activity",
        "digital_wellness": "📱 Digital Wellness",
        "nutrition": "🥗 Nutrition",
        "work_life_balance": "⚖️ Work-Life Balance",
        "maintenance": "✨ Wellness Maintenance",
        "preventive_care": "🛡️ Preventive Care"
    }
    
    cols = st.columns(min(len(plan["focus_areas"]), 4))
    for i, area in enumerate(plan["focus_areas"]):
        with cols[i % 4]:
            st.info(focus_map.get(area, area))
    
    st.markdown("### 📊 Weekly Goals")
    
    goals_df = pd.DataFrame(plan["weekly_goals"])
    if not goals_df.empty:
        for _, row in goals_df.iterrows():
            st.markdown(f"**{row['goal']}** - Target: {row['target']}")
    
    st.markdown("### 📅 Daily Schedule")
    
    # Create tabs for each day
    days = list(plan["daily_schedule"].keys())
    tabs = st.tabs(days)
    
    for tab, day in zip(tabs, days):
        with tab:
            activities = plan["daily_schedule"][day]
            
            # Group by category
            categories = {}
            for activity in activities:
                cat = activity["category"]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(activity)
            
            # Display activities by category
            for category, acts in categories.items():
                st.markdown(f"#### {category}")
                for act in acts:
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"**{act['time']}**")
                        with col2:
                            st.markdown(f"{act['activity']}")
                            if act['duration'] != "N/A":
                                st.caption(f"Duration: {act['duration']}")
                st.markdown("")
    
    # Tips section
    st.markdown("### 💡 Success Tips")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **Getting Started:**
        - Start small - don't try to change everything at once
        - Set reminders on your phone for key activities
        - Track your progress in a journal or app
        - Be patient with yourself - change takes time
        """)
    
    with tips_col2:
        st.markdown("""
        **Staying Consistent:**
        - Find an accountability partner
        - Celebrate small wins
        - Adjust the plan if something isn't working
        - Focus on progress, not perfection
        """)
    
    # Download option
    st.markdown("### 📥 Save Your Plan")
    
    # Create downloadable text version
    plan_text = "YOUR 7-DAY WELLNESS PLAN\n\n"
    plan_text += "=" * 50 + "\n\n"
    
    plan_text += "FOCUS AREAS:\n"
    for area in plan["focus_areas"]:
        plan_text += f"- {focus_map.get(area, area)}\n"
    plan_text += "\n"
    
    plan_text += "WEEKLY GOALS:\n"
    for goal in plan["weekly_goals"]:
        plan_text += f"- {goal['goal']} (Target: {goal['target']})\n"
    plan_text += "\n"
    
    plan_text += "DAILY SCHEDULE:\n"
    plan_text += "=" * 50 + "\n\n"
    
    for day, activities in plan["daily_schedule"].items():
        plan_text += f"\n{day.upper()}\n"
        plan_text += "-" * 30 + "\n"
        for act in activities:
            plan_text += f"{act['time']} - {act['activity']}"
            if act['duration'] != "N/A":
                plan_text += f" ({act['duration']})"
            plan_text += "\n"
        plan_text += "\n"
    
    st.download_button(
        label="📄 Download Plan as Text File",
        data=plan_text,
        file_name=f"wellness_plan_{st.session_state['user_id']}.txt",
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
# Data loading with realistic synthetic generation
# ----------------------------
st.sidebar.header("📊 Data Management")
uploaded = st.sidebar.file_uploader("Upload mental health dataset (CSV)", type=["csv"])

@st.cache_data(show_spinner=True)
def load_default_sample() -> pd.DataFrame:
    """Generate realistic synthetic mental health data matching the actual dataset structure."""
    rng = np.random.default_rng(7)
    n = 2000
    
    # Generate base demographics
    df = pd.DataFrame({
        "Gender": rng.choice(["male", "female", "non-binary", "prefer not to say"], 
                            size=n, p=[0.48, 0.48, 0.03, 0.01]),
        "Age": rng.integers(18, 70, size=n),
        "Occupation": rng.choice(["it", "healthcare", "education", "engineering", 
                                 "finance", "sales", "other"], size=n),
        "Country": rng.choice(["usa", "india", "uk", "canada", "australia", 
                              "germany", "other"], size=n),
    })
    
    # Create correlated lifestyle factors
    high_stress_jobs = df["Occupation"].isin(["healthcare", "finance", "it"])
    
    # Stress level (higher in certain occupations)
    df["Stress_Level"] = rng.integers(0, 11, size=n)
    df.loc[high_stress_jobs, "Stress_Level"] = np.clip(
        df.loc[high_stress_jobs, "Stress_Level"] + rng.integers(1, 4, high_stress_jobs.sum()),
        0, 10
    )
    
    # Sleep hours - inversely correlated with stress
    base_sleep = rng.normal(7.2, 1.3, size=n)
    stress_penalty = (df["Stress_Level"] - 5) * 0.15
    df["Sleep_Hours"] = np.clip(base_sleep - stress_penalty, 3.0, 12.0).round(1)
    
    # Work hours - correlated with stress
    base_work = rng.integers(20, 80, size=n)
    df["Work_Hours"] = base_work
    df.loc[high_stress_jobs, "Work_Hours"] = np.clip(
        df.loc[high_stress_jobs, "Work_Hours"] + rng.integers(5, 15, high_stress_jobs.sum()),
        20, 80
    )
    
    # Physical activity - inversely correlated with work/stress
    base_activity = rng.integers(0, 15, size=n)
    work_penalty = ((df["Work_Hours"] - 40) * 0.1).clip(0, None).astype(int)
    df["Physical_Activity_Hours"] = np.clip(base_activity - work_penalty, 0, 15)
    
    # Social media - higher with stress (escapism) and age
    age_factor = (40 - df["Age"]) * 0.05
    stress_factor = df["Stress_Level"] * 0.15
    df["Social_Media_Usage"] = np.clip(
        rng.exponential(2.0, size=n) + age_factor.clip(0, None) / 10 + stress_factor / 10,
        0.5, 12.0
    ).round(1)
    
    # Diet quality - worse with high stress
    high_stress = df["Stress_Level"] > 7
    diet_probs = np.ones((n, 3)) * [0.30, 0.45, 0.25]
    diet_probs[high_stress] = [0.15, 0.45, 0.40]
    diet_indices = np.array([rng.choice(3, p=prob) for prob in diet_probs])
    df["Diet_Quality"] = np.array(["healthy", "average", "unhealthy"])[diet_indices]
    
    # Smoking - more common with stress
    smoke_probs = np.ones((n, 4)) * [0.65, 0.18, 0.12, 0.05]
    smoke_probs[high_stress] = [0.45, 0.25, 0.20, 0.10]
    smoke_indices = np.array([rng.choice(4, p=prob) for prob in smoke_probs])
    df["Smoking_Habit"] = np.array(["non-smoker", "occasional smoker", 
                                     "regular smoker", "heavy smoker"])[smoke_indices]
    
    # Alcohol - similar pattern
    alcohol_probs = np.ones((n, 4)) * [0.30, 0.45, 0.18, 0.07]
    alcohol_probs[high_stress] = [0.20, 0.40, 0.28, 0.12]
    alcohol_indices = np.array([rng.choice(4, p=prob) for prob in alcohol_probs])
    df["Alcohol_Consumption"] = np.array(["non-drinker", "social drinker", 
                                           "regular drinker", "heavy drinker"])[alcohol_indices]
    
    # Build realistic risk score using proper coefficients
    risk_score = np.zeros(n)
    
    # Base risk
    risk_score += -2.5
    
    # Age (U-shaped: young adults and elderly at higher risk)
    age_normalized = (df["Age"] - df["Age"].mean()) / df["Age"].std()
    risk_score += 0.15 * age_normalized**2
    
    # Lifestyle factors (main predictors)
    risk_score += 2.5 * (df["Stress_Level"] / 10)  # STRONGEST predictor
    risk_score += 1.8 * ((7.5 - df["Sleep_Hours"]).clip(0, None) / 5)  # Sleep deprivation
    risk_score += 1.2 * ((df["Work_Hours"] - 40).clip(0, None) / 30)  # Overwork
    risk_score += 0.9 * ((3 - df["Physical_Activity_Hours"]).clip(0, None) / 3)  # Low exercise
    risk_score += 1.0 * ((df["Social_Media_Usage"] - 3).clip(0, None) / 8)  # Excessive SM
    
    # Health habits
    risk_score += 0.6 * (df["Diet_Quality"] == "unhealthy").astype(int)
    risk_score += 0.8 * df["Smoking_Habit"].isin(["regular smoker", "heavy smoker"]).astype(int)
    risk_score += 0.7 * (df["Alcohol_Consumption"] == "heavy drinker").astype(int)
    
    # Occupational stress
    risk_score += 0.4 * high_stress_jobs.astype(int)
    
    # Add realistic noise (mental health is complex!)
    risk_score += rng.normal(0, 1.2, size=n)
    
    # Convert to probability
    prob = 1 / (1 + np.exp(-risk_score + 3))
    
    # Generate binary outcome with proper randomness
    df["Mental_Health_Condition"] = (rng.uniform(0, 1, size=n) < prob).astype(int)
    
    # Consultation history (people with conditions more likely to have consulted)
    has_condition = df["Mental_Health_Condition"] == 1
    consult_prob = np.where(has_condition, 0.65, 0.15)
    df["Consultation_History"] = (rng.uniform(0, 1, size=n) < consult_prob).astype(int)
    df["Consultation_History"] = df["Consultation_History"].map({1: "yes", 0: "no"})
    
    # Medication usage (subset of those who consulted)
    has_consulted = df["Consultation_History"] == "yes"
    med_prob = np.where(has_condition & has_consulted, 0.70, 0.05)
    df["Medication_Usage"] = (rng.uniform(0, 1, size=n) < med_prob).astype(int)
    df["Medication_Usage"] = df["Medication_Usage"].map({1: "yes", 0: "no"})
    
    # Convert target to yes/no
    df["Mental_Health_Condition"] = df["Mental_Health_Condition"].map({1: "yes", 0: "no"})
    
    return df

if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("✅ Dataset loaded from upload.")
else:
    df = load_default_sample()
    st.sidebar.info("Using synthetic sample data.")

# Expected columns based on dataset description (excluding Severity to avoid data leakage)
EXPECTED_COLUMNS = [
    "Gender", "Age", "Occupation", "Country", "Consultation_History",
    "Stress_Level", "Sleep_Hours", "Work_Hours", 
    "Physical_Activity_Hours", "Social_Media_Usage", "Diet_Quality",
    "Smoking_Habit", "Alcohol_Consumption", "Medication_Usage",
    "Mental_Health_Condition"
]

# Check for missing columns and adapt
available_cols = [c for c in EXPECTED_COLUMNS if c in df.columns]

TARGET_COL = "Mental_Health_Condition"

# Use only available columns
df = df[available_cols].copy()

# Define column types (excluding Severity)
CAT_COLS = [c for c in ["Gender", "Occupation", "Country", "Consultation_History",
                         "Diet_Quality", "Smoking_Habit", 
                         "Alcohol_Consumption", "Medication_Usage"] if c in df.columns]

NUM_COLS = [c for c in ["Age", "Stress_Level", "Sleep_Hours", "Work_Hours",
                         "Physical_Activity_Hours", "Social_Media_Usage"] if c in df.columns]

# Clean data
for c in CAT_COLS + [TARGET_COL]:
    df[c] = df[c].astype(str).str.strip().str.lower()

# Map target to binary
if df[TARGET_COL].isin(["yes", "no"]).any():
    df[TARGET_COL] = df[TARGET_COL].map({"yes": 1, "no": 0}).fillna(0).astype(int)
else:
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

# Numeric columns
for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

# Categorical columns
for c in CAT_COLS:
    df[c] = df[c].astype("category")

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
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int)

    # Split with stratification if possible
    if y.nunique() < 2 or y.value_counts().min() < 10:
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

    cat_categories = {c: list(X_tr[c].cat.categories) for c in CAT_COLS}

    def fit_lgbm(Xdf, y, seed):
        clf = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            num_leaves=31, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbose=-1
        )
        clf.fit(Xdf, y, categorical_feature=CAT_COLS,
                eval_set=[(Xdf, y)],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        return clf

    seeds = [11, 22, 33, 44, 55]
    models = [fit_lgbm(X_tr, y_tr, s) for s in seeds]

    calibrators = []
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

    scaler = StandardScaler().fit(X_tr[NUM_COLS])
    Xtr_scaled = scaler.transform(X_tr[NUM_COLS])
    ood = IsolationForest(n_estimators=300, contamination=0.05, random_state=7)
    ood.fit(Xtr_scaled)

    ranges = {}
    for c in NUM_COLS:
        vals = X_tr[c].dropna()
        if len(vals) > 0:
            lo, hi = np.percentile(vals, [2, 98])
            if hi <= lo:
                hi = lo + 1.0
            ranges[c] = [float(lo), float(hi)]
        else:
            ranges[c] = [0.0, 1.0]

    explainer = shap.TreeExplainer(models[0])

    return Artifacts(models, calibrators, q, eps, scaler, ood, 
                    cat_categories, ranges, explainer, feature_cols)

with st.spinner("🔄 Training model..."):
    arts = train_all(df)

st.sidebar.success("✅ Model trained successfully")

# ----------------------------
# Inference helpers
# ----------------------------
def _prepare_df_row(d: Dict) -> pd.DataFrame:
    row = {k: d.get(k, None) for k in arts.feature_cols}
    X = pd.DataFrame([row])

    for c in NUM_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            X[c] = X[c].fillna(df[c].median())

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

    return X[arts.feature_cols]

def predict_patient(p: Dict) -> Dict:
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

    Xs = X[NUM_COLS].copy()
    score = arts.ood.decision_function(arts.scaler.transform(Xs))[0]
    ood_flag = bool(score < -0.1)

    aleatoric_std = float(np.sqrt(max(aleatoric, 0.0)))

    return {
        "risk": p_mean,
        "uncertainty": {
            "lower": lo,
            "upper": hi,
            "aleatoric_std": aleatoric_std,
            "epistemic_std": p_std,
        },
        "drivers": drivers,
        "ood_flag": ood_flag
    }

# ----------------------------
# User input form
# ----------------------------
st.markdown("---")
st.header("📝 Patient Information Input")

with st.form("patient_form"):
    st.subheader("Demographic Information")
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", options=arts.cat_categories["Gender"])
        age = st.number_input("Age", min_value=18, max_value=200, value=30)
        occupation = st.selectbox("Occupation", options=arts.cat_categories["Occupation"])
    
    with col2:
        country = st.selectbox("Country", options=arts.cat_categories["Country"])
        consultation = st.selectbox("Consultation History", 
                                    options=arts.cat_categories["Consultation_History"])
    
    st.subheader("Lifestyle Factors")
    col3, col4 = st.columns(2)
    
    with col3:
        stress = st.slider("Stress Level", 0, 10, 5, 
                          help="0 = No stress, 10 = Extreme stress")
        sleep = st.slider("Sleep Hours (per night)", 3.0, 12.0, 7.0, 0.5)
        work = st.slider("Work Hours (per week)", 0, 80, 40)
    
    with col4:
        exercise = st.slider("Physical Activity (hours/week)", 0, 20, 3)
        social_media = st.slider("Social Media Usage (hours/day)", 0.0, 12.0, 2.0, 0.5)
        diet = st.selectbox("Diet Quality", options=arts.cat_categories["Diet_Quality"])
    
    st.subheader("Health Habits")
    col5, col6 = st.columns(2)
    
    with col5:
        smoking = st.selectbox("Smoking Habit", options=arts.cat_categories["Smoking_Habit"])
        alcohol = st.selectbox("Alcohol Consumption", 
                              options=arts.cat_categories["Alcohol_Consumption"])
    
    with col6:
        medication = st.selectbox("Medication Usage", 
                                 options=arts.cat_categories["Medication_Usage"])
    
    submitted = st.form_submit_button("🔍 Get Risk Assessment", type="primary", 
                                     use_container_width=True)

if submitted:
    patient_data = {
        "Gender": gender,
        "Age": age,
        "Occupation": occupation,
        "Country": country,
        "Consultation_History": consultation,
        "Stress_Level": stress,
        "Sleep_Hours": sleep,
        "Work_Hours": work,
        "Physical_Activity_Hours": exercise,
        "Social_Media_Usage": social_media,
        "Diet_Quality": diet,
        "Smoking_Habit": smoking,
        "Alcohol_Consumption": alcohol,
        "Medication_Usage": medication
    }
    
    pred = predict_patient(patient_data)
    st.session_state["current_prediction"] = pred
    st.session_state["current_patient"] = patient_data
    
    st.session_state["interaction_log"].append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "event": "prediction_requested",
        "user_id": st.session_state["user_id"],
        "group": st.session_state["study_mode"],
        "patient_data": patient_data,
        "prediction": pred
    })

# ----------------------------
# Results display
# ----------------------------
if "current_prediction" in st.session_state:
    st.markdown("---")
    st.header("📊 Risk Assessment Results")
    
    pred = st.session_state["current_prediction"]
    risk = pred["risk"]
    
    # G1: Basic display (no uncertainty, no explanations)
    if st.session_state["study_mode"] == "G1":
        st.markdown("### Your Mental Health Risk Score")
        
        # Simple risk gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk * 100,
            title = {'text': "Risk Level (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        if risk < 0.33:
            st.success("✅ Low Risk: Your assessment indicates a low risk for mental health conditions.")
        elif risk < 0.66:
            st.warning("⚠️ Moderate Risk: Your assessment indicates moderate risk. Consider consulting with a mental health professional.")
        else:
            st.error("🚨 High Risk: Your assessment indicates elevated risk. We recommend consulting with a mental health professional soon.")
    
    # G2: Enhanced display (with uncertainty and explanations)
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Risk Assessment with Uncertainty")
            
            # Enhanced gauge with uncertainty band
            lo = pred["uncertainty"]["lower"]
            hi = pred["uncertainty"]["upper"]
            
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode = "gauge+number+delta",
                value = risk * 100,
                title = {'text': "Risk Level (%)"},
                delta = {'reference': 50, 'increasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgreen"},
                        {'range': [33, 66], 'color': "yellow"},
                        {'range': [66, 100], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Uncertainty metrics
            st.markdown("#### Prediction Confidence")
            unc_width = hi - lo
            c1, c2, c3 = st.columns(3)
            c1.metric("Lower Bound", f"{lo*100:.1f}%")
            c2.metric("Upper Bound", f"{hi*100:.1f}%")
            c3.metric("Uncertainty Width", f"±{unc_width*50:.1f}%")
            
            # Uncertainty breakdown
            st.markdown("#### Uncertainty Sources")
            epi = pred["uncertainty"]["epistemic_std"]
            alea = pred["uncertainty"]["aleatoric_std"]
            
            col_unc1, col_unc2 = st.columns(2)
            
            with col_unc1:
                st.metric(
                    "🎲 Aleatoric Uncertainty (Data)", 
                    f"{alea:.3f}",
                    help="Inherent randomness - cannot be reduced"
                )
                st.caption("Represents irreducible randomness in outcomes")
            
            with col_unc2:
                st.metric(
                    "🤖 Epistemic Uncertainty (Model)", 
                    f"{epi:.3f}",
                    help="Model uncertainty - can be reduced with more data"
                )
                st.caption("Represents model's confidence level")
            
            # Visual breakdown
            unc_df = pd.DataFrame({
                "Type": ["Aleatoric\n(Data)", "Epistemic\n(Model)"],
                "Value": [alea, epi],
                "Category": ["Data Uncertainty", "Model Uncertainty"]
            })
            
            fig_unc = px.bar(
                unc_df, 
                x="Type", 
                y="Value", 
                color="Category",
                title="Uncertainty Breakdown",
                color_discrete_map={
                    "Data Uncertainty": "#e74c3c",
                    "Model Uncertainty": "#3498db"
                }
            )
            fig_unc.update_layout(showlegend=False, height=250)
            st.plotly_chart(fig_unc, use_container_width=True)
            
            with st.expander("ℹ️ Understanding Uncertainty Types"):
                st.markdown("""
                **Aleatoric Uncertainty (Data Uncertainty):**
                - Inherent randomness in mental health outcomes
                - Cannot be reduced even with more data
                - Reflects natural variability in human behavior
                
                **Epistemic Uncertainty (Model Uncertainty):**
                - Model's lack of knowledge
                - Can be reduced by collecting more training data
                - Higher when model hasn't seen similar cases
                
                **Total Uncertainty = √(Aleatoric² + Epistemic²)**
                """)
            
            if pred.get("ood_flag"):
                st.warning("⚠️ **Note:** Your profile is unusual compared to typical cases. Prediction may be less reliable.")
        
        with col2:
            st.markdown("### AI Explanation: Key Factors")
            st.markdown("These factors had the most impact on your risk assessment:")
            
            # SHAP explanation
            drivers = pred["drivers"]
            driver_df = pd.DataFrame(drivers, columns=["Feature", "Impact"])
            driver_df["Direction"] = driver_df["Impact"].apply(
                lambda x: "Increases Risk ↑" if x > 0 else "Decreases Risk ↓"
            )
            
            # Create horizontal bar chart
            fig_shap = px.bar(
                driver_df,
                y="Feature",
                x="Impact",
                color="Direction",
                orientation='h',
                title="Feature Impact on Risk Prediction",
                color_discrete_map={
                    "Increases Risk ↑": "#ff6b6b",
                    "Decreases Risk ↓": "#51cf66"
                }
            )
            fig_shap.update_layout(height=350, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_shap, use_container_width=True)
            
            with st.expander("📖 Understanding the Factors"):
                for feat, impact in drivers[:3]:
                    direction = "increasing" if impact > 0 else "reducing"
                    st.markdown(f"**{feat}:** This factor is {direction} your risk score.")
        
        # Risk interpretation
        st.markdown("### 🎯 Risk Interpretation")
        if risk < 0.33:
            st.success(f"""
            ✅ **Low Risk** (Uncertainty: {(1-unc_width)*100:.0f}%)
            
            Your assessment indicates low risk for mental health conditions. The model is fairly confident about this prediction.
            """)
        elif risk < 0.66:
            st.warning(f"""
            ⚠️ **Moderate Risk** (Uncertainty: {(1-unc_width)*100:.0f}%)
            
            Your assessment indicates moderate risk. Consider consulting with a mental health professional.
            """)
        else:
            st.error(f"""
            🚨 **High Risk** (Uncertainty: {(1-unc_width)*100:.0f}%)
            
            Your assessment indicates elevated risk. We recommend consulting with a mental health professional soon.
            """)
    
    # ----------------------------
    # Wellness plan generation
    # ----------------------------
    st.markdown("---")
    st.header("📅 Your Personalized 7-Day Wellness Plan")
    
    if st.button("Generate My Wellness Plan", type="primary", use_container_width=True):
        patient_data = st.session_state["current_patient"]
        plan = generate_wellness_plan(patient_data, pred)
        st.session_state["wellness_plan"] = plan
        
        st.session_state["interaction_log"].append({
            "timestamp": pd.Timestamp.now().isoformat(),
            "event": "wellness_plan_generated",
            "user_id": st.session_state["user_id"],
            "group": st.session_state["study_mode"]
        })
    
    if "wellness_plan" in st.session_state:
        display_wellness_plan(st.session_state["wellness_plan"])
        
        # Trust survey
        st.markdown("---")
        st.header("📋 Quick Feedback")
        
        with st.form("trust_survey"):
            st.markdown("Please help us improve by answering these questions:")
            
            trust = st.slider(
                "How much do you trust this AI assessment?",
                1, 5, 3,
                help="1 = Do not trust at all, 5 = Trust completely"
            )
            
            follow = st.slider(
                "How likely are you to follow the wellness plan recommendations?",
                1, 5, 3,
                help="1 = Very unlikely, 5 = Very likely"
            )
            
            useful = st.slider(
                "How useful did you find this assessment?",
                1, 5, 3,
                help="1 = Not useful at all, 5 = Very useful"
            )
            
            comments = st.text_area(
                "Any additional comments? (Optional)",
                placeholder="Share your thoughts about the assessment..."
            )
            
            survey_submit = st.form_submit_button("Submit Feedback", use_container_width=True)
        
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
            
            # Save interaction log
            save_interaction_log()

# ----------------------------
# Admin/Researcher view
# ----------------------------
if st.sidebar.checkbox("🔬 Researcher View (Admin Only)", value=False):
    st.sidebar.markdown("---")
    st.markdown("## 📊 Researcher Dashboard")
    
    log_dir = Path("study_logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("*.json"))
        
        if log_files:
            st.markdown(f"### Collected Data: {len(log_files)} sessions")
            
            # Aggregate data
            all_data = []
            for log_file in log_files:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    all_data.extend(data)
            
            df_logs = pd.DataFrame(all_data)
            
            if not df_logs.empty and 'group' in df_logs.columns:
                # Group statistics
                st.markdown("#### Participation by Group")
                group_counts = df_logs[df_logs['event'] == 'session_start']['group'].value_counts()
                st.bar_chart(group_counts)
                
                # Survey responses
                survey_data = df_logs[df_logs['event'] == 'survey_completed']
                if not survey_data.empty:
                    st.markdown("#### Trust Scores by Group")
                    
                    trust_by_group = survey_data.groupby('group')[['trust_score', 'follow_likelihood', 'usefulness']].mean()
                    st.dataframe(trust_by_group)
                    
                    # Visualize
                    fig = px.bar(
                        trust_by_group.reset_index().melt(id_vars='group'),
                        x='variable',
                        y='value',
                        color='group',
                        barmode='group',
                        title="Average Scores by Group",
                        labels={'variable': 'Metric', 'value': 'Average Score (1-5)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download all data
                st.markdown("#### Export Data")
                csv_data = df_logs.to_csv(index=False)
                st.download_button(
                    "📥 Download All Session Data (CSV)",
                    data=csv_data,
                    file_name="study_data_export.csv",
                    mime="text/csv"
                )
        else:
            st.info("No session data collected yet.")
    else:
        st.info("No study data directory found.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>This is a research tool for educational purposes. Always consult healthcare professionals for medical advice.</p>
    <p>Your data is anonymized and used only for research purposes.</p>
</div>
""", unsafe_allow_html=True)