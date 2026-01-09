import os
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from flask import Flask, render_template, request, redirect, url_for, session, flash
import psycopg2
import psycopg2.extras
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ----------------------------
# Flask setup
# ----------------------------
app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")
app.permanent_session_lifetime = timedelta(minutes=5)
DATABASE_URL = os.environ.get("DATABASE_URL")

# ----------------------------
# Data + model training (once)
# ----------------------------
DATA_PATH = Path("data/stress_risk_data.csv")

def load_default_sample() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

df = load_default_sample()

NUM_COLS = [c for c in df.columns if c not in ["user_id", "stress_risk_score",
                                               "stress_level_category", "data_split", "gender", "occupation"]]
CAT_COLS = ["gender", "occupation"]
TARGET_COL = "stress_risk_score"

# Clean data
df = df[[c for c in df.columns if c not in ["user_id", "data_split"]]].copy()

for c in CAT_COLS:
    if c in df.columns:
        df[c] = (
            df[c]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": np.nan})
        )
        df[c] = df[c].astype("category")

for c in NUM_COLS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

if TARGET_COL in df.columns:
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df["target_binary"] = (df[TARGET_COL] > df[TARGET_COL].median()).astype(int)

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

def train_all(df: pd.DataFrame) -> Artifacts:
    feature_cols = [c for c in NUM_COLS + CAT_COLS if c in df.columns]
    X = df[feature_cols].copy()

    for c in [col for col in CAT_COLS if col in X.columns]:
        if not pd.api.types.is_categorical_dtype(X[c]):
            X[c] = X[c].astype("category")

    y = df["target_binary"].astype(int)

    if y.nunique() < 2 or y.value_counts().min() < 5:
        X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
        X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    else:
        X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=42)
        X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    for frame in (X_tr, X_val, X_te):
        for c in [col for col in CAT_COLS if col in frame.columns]:
            if not pd.api.types.is_categorical_dtype(frame[c]):
                frame[c] = frame[c].astype("category")

    cat_categories: Dict[str, List[str]] = {}
    for c in CAT_COLS:
        if c in X_tr.columns:
            cats = list(pd.Categorical(X_tr[c]).categories)
            if "unknown" not in cats:
                cats = cats + ["unknown"]
            X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
            X_val[c] = pd.Categorical(X_val[c], categories=cats)
            X_te[c] = pd.Categorical(X_te[c], categories=cats)
            cat_categories[c] = cats

    def fit_lgbm(Xdf, y, seed):
        cat_feats = [c for c in CAT_COLS if c in Xdf.columns]
        clf = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1
        )
        clf.fit(
            Xdf, y,
            categorical_feature=cat_feats,
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

    num_feats_present = [c for c in NUM_COLS if c in X_tr.columns]
    if len(num_feats_present) == 0:
        scaler = StandardScaler().fit(np.zeros((len(X_tr), 1)))
        Xtr_scaled = scaler.transform(np.zeros((len(X_tr), 1)))
    else:
        scaler = StandardScaler().fit(X_tr[num_feats_present])
        Xtr_scaled = scaler.transform(X_tr[num_feats_present])

    ood = IsolationForest(n_estimators=300, contamination=0.05, random_state=7)
    ood.fit(Xtr_scaled)

    ranges = {}
    for c in NUM_COLS:
        if c in X_tr.columns:
            vals = X_tr[c].dropna()
            if len(vals) > 0:
                lo, hi = np.percentile(vals, [2, 98])
                if hi <= lo:
                    hi = lo + 1.0
                ranges[c] = [float(lo), float(hi)]

    explainer = shap.TreeExplainer(models[0])

    return Artifacts(models, calibrators, q, eps, scaler, ood, cat_categories, ranges, explainer, feature_cols)

ARTS = train_all(df)

# ----------------------------
# Helpers
# ----------------------------
FEATURE_LABELS = {
    "work_hours_per_week": "Weekly work hours",
    "job_satisfaction": "Job satisfaction",
    "workload_rating": "Workload rating",
    "sleep_duration_hr": "Sleep duration",
    "sleep_quality_rating": "Sleep quality",
    "stress_event_count_last_week": "Stress events last week",
    "breaks_per_workday": "Breaks per workday",
    "coffee_intake_cups": "Coffee intake",
    "alcohol_intake_per_week": "Weekly alcohol intake",
    "physical_activity_frequency": "Weekly exercise",
    "screen_time_hr": "Daily screen time",
    "social_interactions_count": "Weekly social interactions",
    "commute_time_min": "Commute time",
    "outdoor_time_hr": "Outdoor time",
    "screen_unlocks_per_day": "Phone unlocks per day",
}

FEATURE_UNITS = {
    "work_hours_per_week": "hrs/week",
    "sleep_duration_hr": "hrs/night",
    "sleep_quality_rating": "/10",
    "job_satisfaction": "/10",
    "workload_rating": "/10",
    "stress_event_count_last_week": "events",
    "breaks_per_workday": "breaks/day",
    "coffee_intake_cups": "cups/day",
    "alcohol_intake_per_week": "drinks/week",
    "physical_activity_frequency": "sessions/week",
    "screen_time_hr": "hrs/day",
    "social_interactions_count": "per week",
    "commute_time_min": "min/day",
    "outdoor_time_hr": "hrs/day",
    "screen_unlocks_per_day": "per day",
}

def friendly_feature_name(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())

def _prepare_df_row(d: Dict) -> pd.DataFrame:
    row = {k: d.get(k, None) for k in ARTS.feature_cols}
    X = pd.DataFrame([row])

    for c in [col for col in NUM_COLS if col in X.columns]:
        if c in df.columns:
            med = df[c].median()
        else:
            med = 0.0
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med)

    for c in [col for col in CAT_COLS if c in X.columns]:
        X[c] = X[c].astype(str).str.strip().str.lower()
        cats = ARTS.cat_categories.get(c, [])
        if cats:
            X[c] = pd.Categorical(X[c], categories=cats)
            if X[c].isna().any():
                X[c] = X[c].fillna("unknown")
        else:
            X[c] = pd.Categorical(X[c])

    return X[ARTS.feature_cols]

def predict_user(p: Dict) -> Dict:
    X = _prepare_df_row(p)
    ps = np.stack([prob_of_one(cal, X) for cal in ARTS.calibrators], axis=0)
    p_mean = float(ps.mean())
    p_std = float(ps.std())
    aleatoric = float(np.mean(ps * (1 - ps)))
    total_scale = float(np.sqrt(max(aleatoric + p_std**2, 0.0) + ARTS.conformal_eps))
    delta = ARTS.conformal_q * total_scale
    lo = max(0.0, p_mean - delta)
    hi = min(1.0, p_mean + delta)

    shap_vals = ARTS.explainer.shap_values(X)
    shap_arr = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    drivers = sorted(
        [(f, float(shap_arr[0, i])) for i, f in enumerate(X.columns)],
        key=lambda t: abs(t[1]), reverse=True
    )[:5]

    num_feats_present = [c for c in ARTS.feature_cols if c in NUM_COLS and c in X.columns]
    if len(num_feats_present) == 0:
        Xs = np.zeros((1, 1))
    else:
        Xs = ARTS.scaler.transform(X[num_feats_present])
    score = ARTS.ood.decision_function(Xs)[0]
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

def interpret_confidence(lower: float, upper: float) -> Dict[str, str]:
    width = upper - lower
    if width <= 0.10:
        level = "High confidence"
        explainer = "We repeatedly land within a narrow band for people like you."
    elif width <= 0.20:
        level = "Moderate confidence"
        explainer = "Your score usually sits inside a medium-sized band; habits can shift it."
    else:
        level = "Needs more information"
        explainer = "We do not see enough consistent data for similar profiles, so we show a wide band."
    return {"level": level, "explanation": explainer, "width": width}

def compute_coverage_signal(width: float, ood_flag: bool) -> Dict[str, str]:
    if ood_flag:
        return {
            "label": "Low coverage",
            "message": "Your answers look uncommon in our training data, so treat this result with extra care.",
            "tone": "warning",
        }
    if width <= 0.12:
        return {
            "label": "High coverage",
            "message": "We have plenty of participants similar to you, so the estimate is well supported.",
            "tone": "success",
        }
    if width <= 0.20:
        return {
            "label": "Moderate coverage",
            "message": "We have some comparable participants, but lifestyle swings can still move the score.",
            "tone": "info",
        }
    return {
        "label": "Limited coverage",
        "message": "Only a few people in the data look similar, so the model adds a wide safety band.",
        "tone": "warning",
    }

def describe_driver(feature: str, impact: float, user_data: Dict, ranges: Dict[str, List[float]]) -> str:
    name = friendly_feature_name(feature)
    direction = "raises" if impact > 0 else "reduces"
    value = user_data.get(feature, "N/A")
    if isinstance(value, (int, float)):
        lo, hi = ranges.get(feature, (None, None))
        qualifier = ""
        if lo is not None and hi is not None:
            midpoint = (lo + hi) / 2
            if value >= hi:
                qualifier = "- among the highest values we usually see"
            elif value <= lo:
                qualifier = "- among the lowest values we usually see"
            elif value >= midpoint:
                qualifier = "- higher than most participants"
            else:
                qualifier = "- lower than most participants"
        unit = FEATURE_UNITS.get(feature, "")
        formatted_value = f"{value:.1f}".rstrip("0").rstrip(".")
        if unit:
            formatted_value = f"{formatted_value} {unit}"
        return f"{name} {direction} your risk because you reported {formatted_value} {qualifier}".strip()
    return f"{name} {direction} your risk because you reported '{value}'."

def generate_stress_management_plan(user_data: Dict, prediction: Dict) -> Dict:
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
    if not stress_factors:
        stress_factors = ["maintenance", "prevention"]

    plan = {"risk_factors": stress_factors, "daily_routine": {}}
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for i, day in enumerate(days):
        activities = []
        activities.append({
            "time": "6:30-7:00 AM",
            "activity": "Wake up routine - hydrate, light stretching or yoga",
            "category": "Morning",
            "duration": "30 min"
        })
        activities.append({
            "time": "7:00-8:00 AM",
            "activity": "Healthy breakfast - avoid excessive coffee, plan priorities",
            "category": "Breakfast",
            "duration": "60 min"
        })
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
        if "low_exercise" in stress_factors:
            if i < 5:
                activities.append({
                    "time": "5:30 PM" if i % 2 == 0 else "6:30 AM (before work)",
                    "activity": "Moderate exercise - 30-45 min (walk, gym, cycling, yoga)",
                    "category": "Exercise",
                    "duration": "45 min"
                })
            else:
                activities.append({
                    "time": "9:00 AM",
                    "activity": "Longer activity - 60 min outdoor activity or sports",
                    "category": "Exercise",
                    "duration": "60 min"
                })
        if "low_social_contact" in stress_factors:
            if i in [2, 5]:
                activities.append({
                    "time": "7:00 PM",
                    "activity": f"Social time - {'meet friend for coffee' if i == 2 else 'family dinner or video call'}",
                    "category": "Social",
                    "duration": "60 min"
                })
        if "high_caffeine" in stress_factors:
            activities.append({
                "time": "Before 2 PM",
                "activity": "Limit coffee to 1-2 cups in morning (cut off by 2 PM)",
                "category": "Nutrition",
                "duration": "Ongoing"
            })
        if "excessive_screen_time" in stress_factors:
            activities.append({
                "time": "8:30-9:00 PM",
                "activity": "Digital sunset - put phone away, reduce screen exposure",
                "category": "Digital Wellness",
                "duration": "30 min"
            })
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

# ----------------------------
# Logging helpers
# ----------------------------
ACTIVE_LOGS: Dict[str, List[Dict]] = {}
DB_INITIALIZED = False

def get_db_conn():
    if not DATABASE_URL:
        return None
    return psycopg2.connect(DATABASE_URL)

def init_db():
    global DB_INITIALIZED
    if DB_INITIALIZED or not DATABASE_URL:
        return
    conn = get_db_conn()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS study_logs (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT,
                        group_name TEXT,
                        event_type TEXT,
                        event_time TIMESTAMPTZ,
                        payload JSONB
                    );
                """)
        DB_INITIALIZED = True
    finally:
        conn.close()

def get_log() -> List[Dict]:
    uid = session.get("user_id")
    if not uid:
        return []
    return ACTIVE_LOGS.setdefault(uid, [])

def extract_numeric_user_id(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    import re
    match = re.search(r"\d+", str(value))
    if match:
        try:
            return int(match.group())
        except ValueError:
            return None
    return None

def load_last_user_id_from_logs() -> int:
    if DATABASE_URL:
        init_db()
        conn = get_db_conn()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT MAX(NULLIF(regexp_replace(user_id, '\\D', '', 'g'), '')::int)
                        FROM study_logs
                        WHERE user_id IS NOT NULL;
                    """)
                    row = cur.fetchone()
                    if row and row[0]:
                        return int(row[0])
            except Exception:
                pass
            finally:
                conn.close()
    log_dir = Path("study_logs")
    if not log_dir.exists():
        return 0
    max_id = 0
    for log_file in log_dir.glob("*.json"):
        try:
            with open(log_file, "r") as f:
                entries = json.load(f)
            for entry in entries:
                num = extract_numeric_user_id(entry.get("user_id"))
                if num:
                    max_id = max(max_id, num)
        except Exception:
            pass
        stem_first = log_file.stem.split("_")[0]
        num = extract_numeric_user_id(stem_first)
        if num:
            max_id = max(max_id, num)
    return max_id

def ensure_user_session():
    if "user_id" not in session:
        next_id = load_last_user_id_from_logs() + 1
        session["user_id"] = str(next_id)
        group = "G2" if next_id % 2 == 0 else "G1"
        session["group"] = group
        session["study_mode"] = group
        get_log().append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "session_start",
            "user_id": session["user_id"],
            "group": group,
            "assignment": "auto_counter"
        })

def track_page_duration(current_page: str):
    prev_page = session.get("current_page")
    entry_time = session.get("page_entry_time")
    now = datetime.utcnow()
    if prev_page and entry_time and current_page != prev_page:
        try:
            started = datetime.fromisoformat(entry_time)
            duration_seconds = float((now - started).total_seconds())
            get_log().append({
                "timestamp": now.isoformat(),
                "event": "page_duration",
                "page": prev_page,
                "duration_seconds": duration_seconds,
                "user_id": session.get("user_id"),
                "group": session.get("group"),
            })
        except Exception:
            pass
    if current_page != prev_page:
        session["current_page"] = current_page
        session["page_entry_time"] = now.isoformat()

def save_interaction_log():
    entries = get_log()
    if not entries:
        return
    if DATABASE_URL:
        init_db()
        conn = get_db_conn()
        if conn:
            try:
                records = []
                for entry in entries:
                    event_time = entry.get("timestamp")
                    event_dt = None
                    if event_time:
                        try:
                            event_dt = datetime.fromisoformat(event_time)
                        except ValueError:
                            event_dt = None
                    records.append((
                        entry.get("user_id"),
                        entry.get("group"),
                        entry.get("event"),
                        event_dt,
                        psycopg2.extras.Json(entry)
                    ))
                with conn:
                    with conn.cursor() as cur:
                        psycopg2.extras.execute_values(
                            cur,
                            """
                            INSERT INTO study_logs
                                (user_id, group_name, event_type, event_time, payload)
                            VALUES %s
                            """,
                            records
                        )
                return
            except Exception as exc:
                print(f"Failed to save log to Postgres: {exc}")
            finally:
                conn.close()
    log_dir = Path("study_logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{session.get('user_id', 'user')}_{session.get('group', 'G')}_{timestamp}.json"
    log_path = log_dir / filename
    try:
        with open(log_path, "w") as f:
            json.dump(entries, f, indent=2)
    except Exception as exc:
        print(f"Failed to save log: {exc}")

# ----------------------------
# Flask hooks
# ----------------------------
@app.before_request
def before_request():
    if request.endpoint == "static":
        return
    now = datetime.utcnow()
    last_activity = session.get("last_activity")
    if last_activity:
        try:
            last_seen = datetime.fromisoformat(last_activity)
            if now - last_seen > app.permanent_session_lifetime:
                prev_user = session.get("user_id")
                if prev_user:
                    ACTIVE_LOGS.pop(prev_user, None)
                session.clear()
        except ValueError:
            prev_user = session.get("user_id")
            if prev_user:
                ACTIVE_LOGS.pop(prev_user, None)
            session.clear()
    session.permanent = True
    session["last_activity"] = now.isoformat()
    ensure_user_session()
    track_page_duration(request.endpoint or "unknown")

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return redirect(url_for("assessment"))

@app.route("/reset-session")
def reset_session():
    """Clear current participant session so a new user_id/group is assigned."""
    session.clear()
    flash("Session reset. Next request will assign a new participant ID.")
    return redirect(url_for("assessment"))

def parse_form_value(name: str, cast_fn, default=None):
    try:
        return cast_fn(request.form.get(name, default))
    except Exception:
        return default

@app.route("/assessment", methods=["GET", "POST"])
def assessment():
    group = session.get("group", "G1")
    defaults = session.get("form_defaults", {})
    if request.method == "POST":
        action = request.form.get("action")
        if action == "plan":
            pred = session.get("current_prediction")
            user_data = session.get("current_user")
            if not pred or not user_data:
                flash("Submit the assessment first to generate a plan.")
                return redirect(url_for("assessment"))

            plan = generate_stress_management_plan(user_data, pred)
            session["stress_plan"] = plan
            get_log().append({
                "timestamp": datetime.utcnow().isoformat(),
                "event": "plan_generated",
                "user_id": session.get("user_id"),
                "group": group
            })
            return redirect(url_for("assessment") + "#plan")

        age = parse_form_value("age", int, 30)
        gender = request.form.get("gender", "female")
        occupation = request.form.get("occupation", "engineer")
        work_hours = parse_form_value("work_hours_per_week", int, 40)
        job_satisfaction = parse_form_value("job_satisfaction", int, 5)
        workload = parse_form_value("workload_rating", int, 5)
        stress_events = parse_form_value("stress_event_count_last_week", int, 2)
        breaks = parse_form_value("breaks_per_workday", int, 3)
        commute = parse_form_value("commute_time_min", int, 30)
        outdoor_time = parse_form_value("outdoor_time_hr", float, 1.0)
        sleep_quality = parse_form_value("sleep_quality_rating", int, 7)
        sleep_duration = parse_form_value("sleep_duration_hr", float, 7.0)
        exercise = parse_form_value("physical_activity_frequency", int, 3)
        coffee = parse_form_value("coffee_intake_cups", float, 2.0)
        alcohol = parse_form_value("alcohol_intake_per_week", int, 2)
        screen_time = parse_form_value("screen_time_hr", float, 5.0)
        social = parse_form_value("social_interactions_count", int, 8)
        screen_unlocks = parse_form_value("screen_unlocks_per_day", int, 80)

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
        demographics = {"age": age, "gender": gender, "occupation": occupation}

        pred = predict_user(user_data)
        session["current_prediction"] = pred
        session["current_user"] = user_data
        session["current_demographics"] = demographics
        session["form_defaults"] = {**user_data}

        get_log().append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "prediction_requested",
            "user_id": session.get("user_id"),
            "group": group,
            "user_data": user_data,
            "prediction": pred,
            "demographics": demographics
        })
        session.pop("stress_plan", None)
        return redirect(url_for("assessment") + "#results")

    gender_choices = ARTS.cat_categories.get("gender", ["male", "female", "other"]) or ["male", "female", "other"]
    occupation_choices = ARTS.cat_categories.get("occupation", ["engineer", "nurse", "student"]) or ["engineer", "nurse", "student"]
    pred = session.get("current_prediction")
    user_data = session.get("current_user")
    plan = session.get("stress_plan")
    chart_data = None
    driver_descriptions: List[str] = []
    confidence = coverage = None
    risk = None
    drivers = []
    if pred and user_data:
        risk = pred["risk"]
        lo = pred["uncertainty"]["lower"]
        hi = pred["uncertainty"]["upper"]
        confidence = interpret_confidence(lo, hi)
        coverage = compute_coverage_signal(hi - lo, pred.get("ood_flag", False))
        drivers = [(friendly_feature_name(f), float(v)) for f, v in pred.get("drivers", [])]
        driver_descriptions = [describe_driver(f, v, user_data, ARTS.ranges) for f, v in pred.get("drivers", [])]
        chart_data = {
            "risk_pct": risk * 100,
            "lower_pct": lo * 100,
            "upper_pct": hi * 100,
            "drivers": [{"name": d[0], "impact": d[1]} for d in drivers],
            "uncertainty": {
                "aleatoric": float(pred["uncertainty"].get("aleatoric_std", 0.0)) * 100,
                "epistemic": float(pred["uncertainty"].get("epistemic_std", 0.0)) * 100,
            },
        }

    return render_template(
        "assessment.html",
        group=group,
        gender_choices=gender_choices,
        occupation_choices=occupation_choices,
        defaults=defaults,
        user_id=session.get("user_id"),
        pred=pred,
        risk=risk,
        confidence=confidence,
        coverage=coverage,
        drivers=drivers,
        driver_descriptions=driver_descriptions,
        chart_data=chart_data,
        plan=plan,
    )

@app.route("/results", methods=["GET", "POST"])
def results():
    return redirect(url_for("assessment"))

LIKERT_CORE: List[Tuple[str, str]] = [
    ("q_trust_assessment", "I trust the stress risk assessment provided by this system."),
    ("q_confident_rely", "I feel confident relying on this system to understand my stress/mental health."),
    ("q_prediction_accurate", "The prediction seems accurate for me."),
    ("q_risk_reflects_level", "The risk score reflects my current stress level."),
    ("q_results_useful", "The results are useful for understanding my stress/mental health."),
    ("q_recommendations_relevant", "The recommendations feel relevant to my situation."),
    ("q_follow_plan", "I would follow the recommended stress-management plan."),
    ("q_use_again_health", "I would use this system again for health-related decisions."),
    ("q_try_plan_this_week", "How likely are you to try the recommended plan this week?"),
    ("q_ux_easy", "The interface was easy to use."),
    ("q_ux_clear_results", "The results were clear."),
    ("q_ux_comfortable", "I felt comfortable interacting with the system."),
]

LIKERT_GAAIS: List[Tuple[str, str]] = [
    ("q_gaais_beneficial", "AI technologies are generally beneficial for society."),
    ("q_gaais_trust_decisions", "AI systems can usually be trusted to make good decisions."),
    ("q_gaais_bias_concern", "I am concerned about AI making incorrect or biased decisions. (reverse-coded)"),
    ("q_gaais_comfort_using", "I feel comfortable using AI tools in my daily life."),
]

LIKERT_G2_UNCERTAINTY: List[Tuple[str, str]] = [
    ("q_uncertainty_helped", "The uncertainty information helped me understand the system’s confidence."),
    ("q_uncertainty_transparent", "Showing uncertainty made the prediction feel more transparent."),
    ("q_uncertainty_preference", "I prefer systems that show uncertainty over single-number outputs."),
    ("q_xai_helped", "The explanations helped me understand the prediction."),
    ("q_xai_clear", "The feature-importance visualization was clear."),
    ("q_xai_increased_trust", "The explanations increased my trust in the result."),
]

@app.route("/survey", methods=["GET", "POST"])
def survey():
    group = session.get("group", "G1")
    pred = session.get("current_prediction")
    if not pred:
        flash("Complete the assessment and view your results before filling out the survey.")
        return redirect(url_for("assessment"))
    if not session.get("stress_plan"):
        flash("Generate and review your personalized plan before taking the survey.")
        return redirect(url_for("assessment") + "#plan")

    driver_options = [friendly_feature_name(f) for f, _ in pred.get("drivers", [])] or ["Workload rating", "Sleep quality", "Stress events last week"]
    driver_options.append("Not sure / prefer not to answer")

    if request.method == "POST":
        responses: Dict[str, object] = {}
        def grab_int(name, default=3):
            try:
                return int(request.form.get(name, default))
            except Exception:
                return default

        responses["q_demo_age"] = grab_int("q_demo_age", 30)
        responses["q_demo_gender"] = request.form.get("q_demo_gender", "")
        responses["q_demo_occupation"] = request.form.get("q_demo_occupation", "")
        responses["q_demo_education"] = request.form.get("q_demo_education", "")
        responses["q_demo_work_in_tech"] = request.form.get("q_demo_work_in_tech", "No")

        responses["q_health_stress_level"] = grab_int("q_health_stress_level", 3)
        responses["q_health_prior_tool_use"] = request.form.get("q_health_prior_tool_use", "No")

        responses["q_ai_knowledge"] = grab_int("q_ai_knowledge", 3)
        responses["q_ai_tool_frequency"] = request.form.get("q_ai_tool_frequency", "Never")
        responses["q_gaais_beneficial"] = grab_int("q_gaais_beneficial", 3)
        responses["q_gaais_trust_decisions"] = grab_int("q_gaais_trust_decisions", 3)
        responses["q_gaais_bias_concern"] = grab_int("q_gaais_bias_concern", 3)
        responses["q_gaais_comfort_using"] = grab_int("q_gaais_comfort_using", 3)
        responses["q_health_app_frequency"] = request.form.get("q_health_app_frequency", "Never")

        for key, _ in LIKERT_CORE:
            responses[key] = grab_int(key, 3)

        responses["q_attention_check_feature"] = request.form.get("q_attention_check_feature", driver_options[0])
        responses["q_system_preference"] = request.form.get("q_system_preference", "Basic")

        responses["q_open_most_useful"] = request.form.get("q_open_most_useful", "")
        responses["q_open_unclear"] = request.form.get("q_open_unclear", "")
        responses["q_open_suggestions"] = request.form.get("q_open_suggestions", "")

        if group == "G2":
            for key, _ in LIKERT_G2_UNCERTAINTY:
                responses[key] = grab_int(key, 3)
            responses["q_open_uncertainty_impact"] = request.form.get("q_open_uncertainty_impact", "")

        get_log().append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "questionnaire_completed",
            "user_id": session.get("user_id"),
            "group": group,
            "responses": responses,
            "demographics": session.get("current_demographics"),
        })
        save_interaction_log()

        # roll over to next participant automatically
        prev_user = session.get("user_id")
        if prev_user:
            ACTIVE_LOGS.pop(prev_user, None)
        session.clear()
        flash("Thank you! Session saved. Next participant has been assigned.")
        return redirect(url_for("assessment"))

    return render_template(
        "survey.html",
        group=group,
        driver_options=driver_options,
        likert_core=LIKERT_CORE,
        likert_gaais=LIKERT_GAAIS,
        likert_g2=LIKERT_G2_UNCERTAINTY if group == "G2" else [],
        demographics=session.get("current_demographics") or {},
        user_id=session.get("user_id")
    )

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
