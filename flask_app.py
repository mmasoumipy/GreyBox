import io
import json
import os
import socket
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
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
app.permanent_session_lifetime = timedelta(minutes=30)
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

OCCUPATION_SEMANTIC_ORDER = [
    "engineer and computer scientist",
    "technician",
    "researcher",
    "professor",
    "teacher",
    "physician",
    "nurse",
    "resident",
    "medical student",
    "student",
    "admin staff",
    "other",
]

def order_occupations(options):
    base = [str(opt).strip().lower() for opt in options if str(opt).strip()]
    seen = set()
    base_unique = []
    for opt in base:
        if opt not in seen:
            base_unique.append(opt)
            seen.add(opt)
    order_index = {name: i for i, name in enumerate(OCCUPATION_SEMANTIC_ORDER)}
    def sort_key(opt):
        if opt == "other":
            return (2, 0, opt)
        if opt in order_index:
            return (0, order_index[opt], opt)
        return (1, 0, opt)
    return sorted(base_unique, key=sort_key)

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
            cats = [c for c in cats if c != "unknown"]
            if "other" not in cats:
                cats = cats + ["other"]
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
        cal = CalibratedClassifierCV(m, method="isotonic", cv=5)
        cal.fit(X_tr, y_tr)
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
                X[c] = X[c].fillna("other")
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

UNCERTAINTY_COPY = {
    "section_title": "Model Confidence & Familiarity",
    "section_intro": "Confidence shows how stable your score is for people with similar answers. "
                     "Model familiarity shows how common your profile is in the data.",
    "slider_left": "Low confidence",
    "slider_right": "High confidence",
    "confidence_helper": "Confidence shows how steady your result is.\nIf your score would stay similar even if you answered again, confidence is higher.",
    "range_helper": "Small day-to-day changes in your habits could shift your score by about ±{half_span}%.",
    "confidence_chip": "Stability",
    "confidence_levels": {
        "high": {
            "label": "High stability",
            "explanation": "Scores for similar people tend to stay in a tight range.",
        },
        "moderate": {
            "label": "Moderate stability",
            "explanation": "Scores usually stay within a mid-sized range; habits can shift it.",
        },
        "low": {
            "label": "Needs more information",
            "explanation": "We see fewer consistent patterns for similar profiles, so the range is wider.",
        },
    },
    "familiarity_chip": "Model familiarity",
    "familiarity_tooltip": "Technical meaning: how well your answers are represented in the training data.",
    "familiarity_levels": {
        "high": {
            "label": "High familiarity",
            "message": "Many participants look similar to you, so the estimate is well supported.",
            "tone": "success",
        },
        "moderate": {
            "label": "Moderate familiarity",
            "message": "We have some comparable participants, but lifestyle swings can still move the score.",
            "tone": "info",
        },
        "low": {
            "label": "Limited familiarity",
            "message": "Only a few people in the data look similar, so this estimate is less certain.",
            "tone": "warning",
        },
    },
    "learn_more_title": "Learn more about uncertainty",
    "learn_more_items": [
        "Aleatoric: natural life variability that cannot be fully reduced.",
        "Epistemic: model uncertainty that can shrink as we collect more similar data.",
    ],
    "variance_title": "What could affect your stress score",
    "variance_intro": "The main factors that could affect your specific score are:",
    "variance_row_aleatoric": "Daily life changes (sleep, mood, unexpected events)",
    "variance_row_epistemic": "Fewer similar profiles in the data",
    "variance_fallback_a": "Some variation comes from real-life factors we can’t fully predict.",
    "variance_fallback_b": "Some variation comes from having fewer highly similar cases in our data.",
    "variance_total": "Your score could vary by about ±{half_span}%.",
}

def interpret_confidence(lower: float, upper: float) -> Dict[str, str]:
    width = upper - lower
    if width <= 0.10:
        level = UNCERTAINTY_COPY["confidence_levels"]["high"]["label"]
        explainer = UNCERTAINTY_COPY["confidence_levels"]["high"]["explanation"]
    elif width <= 0.20:
        level = UNCERTAINTY_COPY["confidence_levels"]["moderate"]["label"]
        explainer = UNCERTAINTY_COPY["confidence_levels"]["moderate"]["explanation"]
    else:
        level = UNCERTAINTY_COPY["confidence_levels"]["low"]["label"]
        explainer = UNCERTAINTY_COPY["confidence_levels"]["low"]["explanation"]
    return {"level": level, "explanation": explainer, "width": width}

def compute_coverage_signal(width: float, ood_flag: bool) -> Dict[str, str]:
    if ood_flag:
        return UNCERTAINTY_COPY["familiarity_levels"]["low"]
    if width <= 0.12:
        return UNCERTAINTY_COPY["familiarity_levels"]["high"]
    if width <= 0.20:
        return UNCERTAINTY_COPY["familiarity_levels"]["moderate"]
    return UNCERTAINTY_COPY["familiarity_levels"]["low"]

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

    def add_activity(day: str, time: str, activity: str, duration: str = "") -> None:
        plan["daily_routine"].setdefault(day, []).append({
            "time": time,
            "activity": activity,
            "duration": duration,
        })

    def generate_daily_routine_with_openai() -> Dict[str, List[Dict[str, str]]]:
        api_key = os.getenv("OPENAI_Grey_API_KEY")
        if not api_key:
            return {}
        prompt = {
            "role": "system",
            "content": (
                "You are a wellness coach. Generate a concise daily plan for Monday-Sunday based on the user's stress factors. "
                "Return JSON only, with a top-level key 'daily_routine' as an object with keys Monday-Sunday. "
                "Each day is a list of items, each item must include: time, activity, duration (strings). "
                "Keep it practical and non-medical."
            ),
        }
        user_msg = {
            "role": "user",
            "content": json.dumps({
                "stress_factors": stress_factors,
                "user_data": user_data,
                "constraints": [
                    "Include movement options like yoga, pilates, or stretching at least 3x/week.",
                    "Use realistic times (morning, lunch, evening).",
                    "Limit to 3-5 items per day.",
                ],
            }),
        }
        payload = {
            "model": "gpt-4o-mini",
            "temperature": 0.4,
            "messages": [prompt, user_msg],
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                body = resp.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, socket.timeout):
            return {}

        try:
            result = json.loads(body)
            content = result["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.startswith("json"):
                    content = content[4:].strip()
            parsed = json.loads(content)
            routine = parsed.get("daily_routine", {})
            if isinstance(routine, dict):
                cleaned = {}
                for day, items in routine.items():
                    if not isinstance(items, list):
                        continue
                    cleaned_items = []
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        cleaned_items.append({
                            "time": str(item.get("time", "")).strip(),
                            "activity": str(item.get("activity", "")).strip(),
                            "duration": str(item.get("duration", "")).strip(),
                        })
                    cleaned[day] = [i for i in cleaned_items if i["activity"]]
                return cleaned
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            return {}
        return {}

    openai_routine = generate_daily_routine_with_openai()
    if openai_routine:
        plan["daily_routine"] = openai_routine
        return plan

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for i, day in enumerate(days):
        add_activity(day, "7:00 AM", "Morning reset - hydrate and plan top priorities", "10 min")
        add_activity(day, "12:30 PM", "Midday break - step outside or short walk", "10 min")
        if "low_exercise" in stress_factors:
            if day in ["Monday", "Wednesday", "Friday"]:
                add_activity(day, "6:00 PM", "Movement - yoga/pilates/stretching", "30-45 min")
        if "high_workload" in stress_factors or "insufficient_breaks" in stress_factors:
            add_activity(day, "3:00 PM", "Micro-break - breathing or stretch", "5 min")
        if day in ["Saturday", "Sunday"]:
            add_activity(day, "10:00 AM", "Longer outdoor activity", "45-60 min")
        if "poor_sleep_quality" in stress_factors or "insufficient_sleep" in stress_factors:
            add_activity(day, "9:30 PM", "Wind-down routine", "30-60 min")

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
            "target": "3x per week (20-45 min)",
            "tips": ["Brisk walking", "Yoga", "Pilates", "Cycling", "Team sports"]
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
    return redirect(url_for("ethics"))

@app.route("/reset-session")
def reset_session():
    """Clear current participant session so a new user_id/group is assigned."""
    session.clear()
    flash("Session reset. Next request will assign a new participant ID.")
    return redirect(url_for("ethics"))

@app.route("/ethics", methods=["GET", "POST"])
def ethics():
    if request.method == "POST":
        session["ethics_ack"] = True
        session["ethics_ack_ts"] = datetime.utcnow().isoformat()
        return redirect(url_for("pre_assessment"))
    return render_template("ethics.html")

def parse_form_value(name: str, cast_fn, default=None):
    try:
        return cast_fn(request.form.get(name, default))
    except Exception:
        return default

@app.route("/pre-assessment", methods=["GET", "POST"])
def pre_assessment():
    if not session.get("ethics_ack"):
        return redirect(url_for("ethics"))
    group = session.get("group", "G1")
    gender_choices = ARTS.cat_categories.get("gender", ["male", "female", "other"]) or ["male", "female", "other"]
    occupation_choices = ARTS.cat_categories.get("occupation", ["engineer", "nurse", "student"]) or ["engineer", "nurse", "student"]
    occupation_choices = [opt for opt in occupation_choices if str(opt).lower() != "unknown"]
    lower_choices = [str(opt).lower() for opt in occupation_choices]
    if "other" not in lower_choices:
        occupation_choices.append("other")
    if "teacher" not in lower_choices:
        occupation_choices.append("teacher")
    if "professor" not in lower_choices:
        occupation_choices.append("professor")
    occupation_choices = order_occupations(occupation_choices)

    defaults = {
        **(session.get("current_demographics") or {}),
        **(session.get("gaais_responses") or {}),
        **(session.get("pre_assessment_defaults") or {}),
    }

    if request.method == "POST":
        required_fields = {
            "age": "Age",
            "gender": "Gender",
            "occupation": "Occupation",
            "education": "Highest education level",
            "field_of_study_work": "Field of study/work",
        }
        for key, label in LIKERT_GAAIS:
            required_fields[key] = label

        missing = [label for key, label in required_fields.items() if not request.form.get(key)]
        if missing:
            for label in missing:
                flash(f"{label} is required.", "error")
            session["pre_assessment_defaults"] = request.form.to_dict()
            defaults = request.form.to_dict()
            return render_template(
                "pre_assessment.html",
                gender_choices=gender_choices,
                occupation_choices=occupation_choices,
                education_options=EDUCATION_OPTIONS,
                yes_no_options=YES_NO_OPTIONS,
                likert_gaais=LIKERT_GAAIS,
                likert_gaais_min=1,
                likert_gaais_max=5,
                likert_gaais_default=3,
                defaults=defaults,
                user_id=session.get("user_id"),
            )

        numeric_constraints = {"age": (int, 18, 90)}
        numeric_errors = []
        parsed_values = {}
        for name, (cast_fn, min_value, max_value) in numeric_constraints.items():
            raw = request.form.get(name, "")
            try:
                value = cast_fn(raw)
            except Exception:
                numeric_errors.append(
                    f"{required_fields[name]} must be a number between {min_value} and {max_value}."
                )
                continue
            if value < min_value or value > max_value:
                numeric_errors.append(
                    f"{required_fields[name]} must be between {min_value} and {max_value}."
                )
                continue
            parsed_values[name] = value

        if numeric_errors:
            for message in numeric_errors:
                flash(message, "error")
            session["pre_assessment_defaults"] = request.form.to_dict()
            defaults = request.form.to_dict()
            return render_template(
                "pre_assessment.html",
                gender_choices=gender_choices,
                occupation_choices=occupation_choices,
                education_options=EDUCATION_OPTIONS,
                yes_no_options=YES_NO_OPTIONS,
                likert_gaais=LIKERT_GAAIS,
                likert_gaais_min=1,
                likert_gaais_max=5,
                likert_gaais_default=3,
                defaults=defaults,
                user_id=session.get("user_id"),
            )

        age = parsed_values["age"]
        gender = request.form.get("gender", "female")
        occupation = request.form.get("occupation", "engineer")
        education = request.form.get("education", "")
        field_of_study_work = request.form.get("field_of_study_work", "").strip()
        demographics = {
            "age": age,
            "gender": gender,
            "occupation": occupation,
            "education": education,
            "field_of_study_work": field_of_study_work,
        }
        gaais_responses = {}
        for key, _ in LIKERT_GAAIS:
            try:
                gaais_responses[key] = int(request.form.get(key, 3))
            except Exception:
                gaais_responses[key] = 3

        session["current_demographics"] = demographics
        session["gaais_responses"] = gaais_responses
        session["pre_assessment_complete"] = True
        session.pop("pre_assessment_defaults", None)

        get_log().append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "pre_assessment_completed",
            "user_id": session.get("user_id"),
            "group": group,
            "demographics": demographics,
            "gaais": gaais_responses,
        })

        return redirect(url_for("assessment"))

    return render_template(
        "pre_assessment.html",
        gender_choices=gender_choices,
        occupation_choices=occupation_choices,
        education_options=EDUCATION_OPTIONS,
        yes_no_options=YES_NO_OPTIONS,
        likert_gaais=LIKERT_GAAIS,
        likert_gaais_min=1,
        likert_gaais_max=5,
        likert_gaais_default=3,
        defaults=defaults,
        user_id=session.get("user_id"),
    )

WEEKS_PER_MONTH = 52 / 12

@app.route("/assessment", methods=["GET", "POST"])
def assessment():
    if not session.get("ethics_ack"):
        return redirect(url_for("ethics"))
    demographics = session.get("current_demographics") or {}
    gaais_responses = session.get("gaais_responses") or {}
    if "age" not in demographics or not gaais_responses:
        return redirect(url_for("pre_assessment"))
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

        required_fields = {
            "work_hours_per_week": "Work Hours / Week",
            "job_satisfaction": "Job Satisfaction",
            "workload_rating": "Workload Rating",
            "stress_event_count_last_week": "Stress Events (Last Week)",
            "breaks_per_workday": "Breaks per Workday",
            "commute_time_min": "Commute Time",
            "outdoor_time_hr": "Outdoor Time",
            "sleep_quality_rating": "Sleep Quality",
            "sleep_duration_hr": "Sleep Duration",
            "physical_activity_frequency": "Physical Activity",
            "coffee_intake_cups": "Coffee",
            "alcohol_intake_per_month": "Alcohol",
            "screen_time_hr": "Screen Time",
            "social_interactions_count": "Social Interactions",
            "screen_unlocks_per_day": "Screen Unlocks / Day",
        }
        numeric_constraints = {
            "work_hours_per_week": (int, 0, 120),
            "job_satisfaction": (int, 1, 10),
            "workload_rating": (int, 1, 10),
            "stress_event_count_last_week": (int, 0, 50),
            "breaks_per_workday": (int, 0, 20),
            "commute_time_min": (int, 0, 300),
            "outdoor_time_hr": (float, 0, 12),
            "sleep_quality_rating": (int, 1, 10),
            "sleep_duration_hr": (float, 0, 16),
            "physical_activity_frequency": (int, 0, 21),
            "coffee_intake_cups": (float, 0, 15),
            "alcohol_intake_per_month": (float, 0, 200),
            "screen_time_hr": (float, 0, 24),
            "social_interactions_count": (int, 0, 50),
            "screen_unlocks_per_day": (int, 0, 500),
        }
        missing = [label for key, label in required_fields.items() if not request.form.get(key)]
        if missing:
            for label in missing:
                flash(f"{label} is required.", "error")
            defaults = request.form.to_dict()
            return render_template(
                "assessment.html",
                group=group,
                defaults=defaults,
                user_id=session.get("user_id"),
                pred=session.get("current_prediction"),
                risk=None,
                confidence=None,
                coverage=None,
                drivers=[],
                driver_descriptions=[],
                chart_data=None,
                plan=session.get("stress_plan"),
                uncertainty_copy=UNCERTAINTY_COPY,
            )

        numeric_errors = []
        parsed_values = {}
        for name, (cast_fn, min_value, max_value) in numeric_constraints.items():
            raw = request.form.get(name, "")
            try:
                value = cast_fn(raw)
            except Exception:
                numeric_errors.append(
                    f"{required_fields[name]} must be a number between {min_value} and {max_value}."
                )
                continue
            if value < min_value or value > max_value:
                numeric_errors.append(
                    f"{required_fields[name]} must be between {min_value} and {max_value}."
                )
                continue
            parsed_values[name] = value

        if numeric_errors:
            for message in numeric_errors:
                flash(message, "error")
            defaults = request.form.to_dict()
            return render_template(
                "assessment.html",
                group=group,
                defaults=defaults,
                user_id=session.get("user_id"),
                pred=session.get("current_prediction"),
                risk=None,
                confidence=None,
                coverage=None,
                drivers=[],
                driver_descriptions=[],
                chart_data=None,
                plan=session.get("stress_plan"),
                uncertainty_copy=UNCERTAINTY_COPY,
            )

        age = demographics.get("age", 30)
        gender = demographics.get("gender", "female")
        occupation = demographics.get("occupation", "engineer")
        education = demographics.get("education", "")
        work_hours = parsed_values["work_hours_per_week"]
        job_satisfaction = parsed_values["job_satisfaction"]
        workload = parsed_values["workload_rating"]
        stress_events = parsed_values["stress_event_count_last_week"]
        breaks = parsed_values["breaks_per_workday"]
        commute = parsed_values["commute_time_min"]
        outdoor_time = parsed_values["outdoor_time_hr"]
        sleep_quality = parsed_values["sleep_quality_rating"]
        sleep_duration = parsed_values["sleep_duration_hr"]
        exercise = parsed_values["physical_activity_frequency"]
        coffee = parsed_values["coffee_intake_cups"]
        alcohol_month = parsed_values["alcohol_intake_per_month"]
        alcohol = alcohol_month / WEEKS_PER_MONTH
        screen_time = parsed_values["screen_time_hr"]
        social = parsed_values["social_interactions_count"]
        screen_unlocks = parsed_values["screen_unlocks_per_day"]

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
        session["current_prediction"] = pred
        session["current_user"] = user_data
        session["form_defaults"] = {
            **user_data,
            "alcohol_intake_per_month": alcohol_month,
        }

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
                "aleatoric": float(pred["uncertainty"].get("aleatoric_std", 0.0)) ** 2,
                "epistemic": float(pred["uncertainty"].get("epistemic_std", 0.0)) ** 2,
            },
        }

    return render_template(
        "assessment.html",
        group=group,
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
        uncertainty_copy=UNCERTAINTY_COPY,
    )

@app.route("/results", methods=["GET", "POST"])
def results():
    return redirect(url_for("assessment"))

LIKERT_CORE: List[Tuple[str, str]] = [
    ("q_tam_pu_useful", "Using this stress assessment system would help me manage my stress more quickly."),
    ("q_tam_pu_effective", "Using this stress assessment system would improve how effectively I manage my stress."),
    ("q_tam_pu_productive", "Using this stress assessment system would increase my productivity in taking care of my wellbeing."),
    ("q_tam_pu_easier", "Using this stress assessment system would make it easier to manage my stress."),
    ("q_tam_pu_understand", "Using this stress assessment system would enhance my effectiveness in monitoring my stress levels."),
    ("q_tam_pu_value", "I would find this stress assessment system useful for my stress-related decisions."),
    ("q_tam_peou_learn", "Learning to use this stress assessment system would be easy for me."),
    ("q_tam_peou_clear", "My interaction with this stress assessment system would be clear and understandable."),
    ("q_tam_peou_skillful", "It would be easy for me to become skillful at using this stress assessment system."),
    ("q_tam_peou_flexible", "I would find this stress assessment system flexible to interact with."),
    ("q_tam_peou_do_what", "I would find it easy to get this stress assessment system to do what I want it to do."),
    ("q_tam_peou_easy", "Overall, I would find this stress assessment system easy to use."),
]

LIKERT_GAAIS: List[Tuple[str, str]] = [
    ("q_gaais_interest", "I am interested in using artificial intelligence systems in my daily life."),
    ("q_gaais_impact", "Artificial Intelligence can have positive impacts on people's wellbeing."),
    ("q_gaais_excitement", "Artificial Intelligence is exciting."),
    ("q_gaais_benefit", "Much of society will benefit from a future full of Artificial Intelligence."),
    ("q_gaais_employment", "I would like to use artificial intelligence in my own job."),
    ("q_gaais_sinister", "I find artificial intelligence sinister."),
    ("q_gaais_control", "Artificial Intelligence might take control of people."),
    ("q_gaais_danger", "I think artificial intelligence is dangerous."),
    ("q_gaais_discomfort", "I shiver with discomfort when I think about future uses of artificial intelligence."),
    ("q_gaais_suffering", "People like me will suffer if artificial intelligence is used more and more."),
]

LIKERT_G2_UNCERTAINTY: List[Tuple[str, str]] = [
    ("q_ess_understand", "From the explanation, I understand how this stress assessment system works."),
    ("q_ess_satisfying", "This explanation of how this stress assessment system works is satisfying."),
    ("q_ess_detail", "This explanation of how this stress assessment system works has sufficient detail."),
    ("q_ess_complete", "This explanation seems complete."),
    ("q_ess_accurate", "This explanation shows how accurate this stress assessment system is."),
    ("q_ess_reliable", "This explanation shows me how reliable this stress assessment system is."),
    ("q_ess_how_to_use", "This explanation tells me how to use this stress assessment system."),
    ("q_ess_useful_goals", "This explanation is useful to my goals."),
    ("q_ess_trust_calibration", "This explanation helps me know when I should trust and not trust this stress assessment system."),
]

EDUCATION_OPTIONS = [
    "High school",
    "Associate degree",
    "Bachelor's degree",
    "Master's degree",
    "Doctorate",
    "Other",
]
YES_NO_OPTIONS = ["No", "Yes"]
AI_TOOL_FREQUENCY_OPTIONS = ["Never", "Occasionally", "Weekly", "Daily"]
HEALTH_APP_FREQUENCY_OPTIONS = ["Never", "Monthly", "Weekly", "Daily"]

@app.route("/survey", methods=["GET"])
def survey():
    return redirect(url_for("survey_step", step=1))

@app.route("/survey/step/<int:step>", methods=["GET", "POST"])
def survey_step(step: int):
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

    total_steps = 3
    if step < 1 or step > total_steps:
        return redirect(url_for("survey_step", step=1))

    survey_form = session.get("survey_form", {})
    validation_labels = {
        "q_health_stress_level": "Self-rated stress level",
        "q_health_prior_tool_use": "Prior tool use",
        "q_ai_knowledge": "AI knowledge",
        "q_ai_tool_frequency": "AI tool use frequency",
        "q_health_app_frequency": "Health app/wearable frequency",
        "q_attention_check_feature": "Attention check",
        "q_open_most_useful": "Most useful part",
        "q_open_unclear": "Unclear part",
        "q_open_suggestions": "Suggestions",
        "q_open_uncertainty_impact": "Uncertainty impact: Did the uncertainty or explanations change how you interpreted the results?",
        "q_open_uncertainty_transparency": "Uncertainty transparency impact: How did the uncertainty information affect your perception of the system’s transparency?",
        "q_open_uncertainty_confidence": "Uncertainty confidence impact: How did the uncertainty visualization influence your understanding of the system’s confidence?",
    }

    def grab_int(name: str, default: int = 3) -> int:
        try:
            return int(request.form.get(name, default))
        except Exception:
            return default

    step_fields = {
        1: [
            "q_health_stress_level",
            "q_health_prior_tool_use",
            "q_ai_knowledge",
            "q_ai_tool_frequency",
            "q_health_app_frequency",
        ],
        2: [],
        3: [
            "q_attention_check_feature",
            "q_open_most_useful",
            "q_open_unclear",
            "q_open_suggestions",
        ],
    }
    if group == "G2":
        step_fields[3].extend([
            "q_open_uncertainty_impact",
            "q_open_uncertainty_transparency",
            "q_open_uncertainty_confidence",
        ])

    if request.method == "POST":
        if "back" in request.form:
            for field in step_fields.get(step, []):
                survey_form[field] = request.form.get(field, "")
            if step == 2:
                for key, _ in LIKERT_CORE:
                    survey_form[key] = request.form.get(key, "")
            if step == 3 and group == "G2":
                for key, _ in LIKERT_G2_UNCERTAINTY:
                    survey_form[key] = request.form.get(key, "")
            session["survey_form"] = survey_form
            return redirect(url_for("survey_step", step=max(1, step - 1)))

        required_fields = list(step_fields.get(step, []))
        if step == 2:
            required_fields.extend([key for key, _ in LIKERT_CORE])
        if step == 3 and group == "G2":
            required_fields.extend([key for key, _ in LIKERT_G2_UNCERTAINTY])

        missing = [validation_labels.get(field, field) for field in required_fields if not request.form.get(field)]
        if missing:
            for label in missing:
                flash(f"{label} is required.", "error")
            defaults = {**survey_form, **request.form.to_dict()}
            return render_template(
                "survey.html",
                group=group,
                driver_options=driver_options,
                likert_core=LIKERT_CORE,
                likert_g2=LIKERT_G2_UNCERTAINTY if group == "G2" else [],
                likert_g2_min=1,
                likert_g2_max=5,
                likert_g2_default=3,
                user_id=session.get("user_id"),
                step=step,
                total_steps=total_steps,
                defaults=defaults,
                education_options=EDUCATION_OPTIONS,
                yes_no_options=YES_NO_OPTIONS,
                ai_tool_frequency_options=AI_TOOL_FREQUENCY_OPTIONS,
                health_app_frequency_options=HEALTH_APP_FREQUENCY_OPTIONS,
            )
        for field in step_fields.get(step, []):
            if field in [
                "q_open_most_useful",
                "q_open_unclear",
                "q_open_suggestions",
                "q_open_uncertainty_impact",
                "q_open_uncertainty_transparency",
                "q_open_uncertainty_confidence",
            ]:
                survey_form[field] = request.form.get(field, "")
            else:
                survey_form[field] = request.form.get(field, "")

        if step == 2:
            for key, _ in LIKERT_CORE:
                survey_form[key] = request.form.get(key, "")
        if step == 3 and group == "G2":
            for key, _ in LIKERT_G2_UNCERTAINTY:
                survey_form[key] = request.form.get(key, "")

        session["survey_form"] = survey_form
        if "next" in request.form and step < total_steps:
            return redirect(url_for("survey_step", step=step + 1))

        if "submit" in request.form and step == total_steps:
            responses: Dict[str, object] = {}
            def to_int(name: str, default: int = 3) -> int:
                try:
                    return int(survey_form.get(name, default))
                except Exception:
                    return default

            responses["q_health_stress_level"] = to_int("q_health_stress_level", 3)
            responses["q_health_prior_tool_use"] = survey_form.get("q_health_prior_tool_use", "No")

            responses["q_ai_knowledge"] = to_int("q_ai_knowledge", 3)
            responses["q_ai_tool_frequency"] = survey_form.get("q_ai_tool_frequency", "Never")
            responses["q_health_app_frequency"] = survey_form.get("q_health_app_frequency", "Never")

            for key, _ in LIKERT_CORE:
                responses[key] = to_int(key, 3)

            responses["q_attention_check_feature"] = survey_form.get("q_attention_check_feature", driver_options[0])

            responses["q_open_most_useful"] = survey_form.get("q_open_most_useful", "")
            responses["q_open_unclear"] = survey_form.get("q_open_unclear", "")
            responses["q_open_suggestions"] = survey_form.get("q_open_suggestions", "")

            if group == "G2":
                for key, _ in LIKERT_G2_UNCERTAINTY:
                    responses[key] = to_int(key, 3)
                responses["q_open_uncertainty_impact"] = survey_form.get("q_open_uncertainty_impact", "")
                responses["q_open_uncertainty_transparency"] = survey_form.get("q_open_uncertainty_transparency", "")
                responses["q_open_uncertainty_confidence"] = survey_form.get("q_open_uncertainty_confidence", "")

            gaais_responses = session.get("gaais_responses") or {}
            for key, _ in LIKERT_GAAIS:
                responses[key] = int(gaais_responses.get(key, 3))

            get_log().append({
                "timestamp": datetime.utcnow().isoformat(),
                "event": "questionnaire_completed",
                "user_id": session.get("user_id"),
                "group": group,
                "responses": responses,
                "demographics": session.get("current_demographics"),
            })
            save_interaction_log()

            prev_user = session.get("user_id")
            if prev_user:
                ACTIVE_LOGS.pop(prev_user, None)
            session.clear()
            return redirect(url_for("survey_thanks"))

    return render_template(
        "survey.html",
        group=group,
        driver_options=driver_options,
        likert_core=LIKERT_CORE,
        likert_g2=LIKERT_G2_UNCERTAINTY if group == "G2" else [],
        likert_g2_min=1,
        likert_g2_max=5,
        likert_g2_default=3,
        user_id=session.get("user_id"),
        step=step,
        total_steps=total_steps,
        defaults=survey_form,
        education_options=EDUCATION_OPTIONS,
        yes_no_options=YES_NO_OPTIONS,
        ai_tool_frequency_options=AI_TOOL_FREQUENCY_OPTIONS,
        health_app_frequency_options=HEALTH_APP_FREQUENCY_OPTIONS,
    )

@app.route("/plan/pdf", methods=["GET"])
def plan_pdf():
    plan = session.get("stress_plan")
    if not plan:
        flash("Generate your personalized plan before downloading the PDF.", "error")
        return redirect(url_for("assessment") + "#plan")

    buffer = io.BytesIO()
    page_width, page_height = letter
    margin = 0.75 * inch
    y = page_height - margin
    c = canvas.Canvas(buffer, pagesize=letter)

    def new_page() -> None:
        nonlocal y
        c.showPage()
        y = page_height - margin

    def ensure_space(needed: float) -> None:
        nonlocal y
        if y - needed < margin:
            new_page()

    def draw_heading(text: str, size: int = 18) -> None:
        nonlocal y
        ensure_space(size + 10)
        c.setFont("Helvetica-Bold", size)
        c.drawString(margin, y, text)
        y -= size + 6

    def draw_subheading(text: str, size: int = 12) -> None:
        nonlocal y
        ensure_space(size + 8)
        c.setFont("Helvetica-Bold", size)
        c.drawString(margin, y, text)
        y -= size + 6

    def draw_paragraph(text: str, size: int = 11, leading: int = 14) -> None:
        nonlocal y
        c.setFont("Helvetica", size)
        for line in textwrap.wrap(text, width=95):
            ensure_space(leading)
            c.drawString(margin, y, line)
            y -= leading
        y -= 4

    def draw_bullets(lines: List[str], size: int = 11, leading: int = 14) -> None:
        nonlocal y
        c.setFont("Helvetica", size)
        for line in lines:
            for wrapped in textwrap.wrap(line, width=90):
                ensure_space(leading)
                c.drawString(margin + 10, y, f"• {wrapped}")
                y -= leading
        y -= 4

    def get_attr(obj, name: str, default=""):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    draw_heading("Personalized Stress Plan")
    draw_paragraph(f"Generated on {datetime.utcnow().strftime('%B %d, %Y')}.")

    risk_factors = plan.get("risk_factors", [])
    if risk_factors:
        draw_subheading("Risk Factors")
        draw_bullets([str(rf).replace("_", " ").title() for rf in risk_factors])

    weekly_goals = plan.get("weekly_goals", [])
    if weekly_goals:
        draw_subheading("Weekly Goals")
        for goal in weekly_goals:
            title = str(get_attr(goal, "goal", "")).title()
            target = str(get_attr(goal, "target", "")).strip()
            tips = get_attr(goal, "tips", [])
            tips_text = ", ".join(tips) if isinstance(tips, list) else str(tips)
            if title:
                draw_paragraph(f"{title}: {target}" if target else title)
            if tips_text:
                draw_bullets([f"Suggestions: {tips_text}"])

    daily_routine = plan.get("daily_routine", {})
    if daily_routine:
        draw_subheading("Daily Routine")
        for day, activities in daily_routine.items():
            draw_paragraph(str(day))
            for act in activities or []:
                time = get_attr(act, "time", "")
                activity = get_attr(act, "activity", "")
                duration = get_attr(act, "duration", "")
                detail = f"{time} – {activity}".strip(" –")
                if duration:
                    detail = f"{detail} ({duration})"
                draw_bullets([detail] if detail else [])

    c.showPage()
    c.save()
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="personalized_plan.pdf",
    )

@app.route("/survey/thanks", methods=["GET"])
def survey_thanks():
    return render_template("survey_thanks.html")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
