import ctypes.util
import sys
from typing import Dict, List
from dataclasses import dataclass
import json
from pathlib import Path
import os
import re
from contextlib import contextmanager

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

try:
    import analyze_study as study_analysis
except Exception:
    study_analysis = None

st.set_page_config(page_title="Burnout Assessment", layout="wide")

st.markdown(
    """
    <style>
    .result-card {
        background: rgba(31, 119, 180, 0.06);
        border: 1px solid rgba(31, 119, 180, 0.2);
        border-radius: 18px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 18px 35px rgba(15, 23, 42, 0.08);
    }
    .result-card__tag {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        color: #1f77b4;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .result-card h3, .result-card h4 {
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@contextmanager
def result_card(title: str | None = None, tag: str | None = None):
    outer = st.container()
    outer.markdown('<div class="result-card">', unsafe_allow_html=True)
    inner = outer.container()
    if tag:
        inner.markdown(f"<p class='result-card__tag'>{tag}</p>", unsafe_allow_html=True)
    if title:
        inner.markdown(f"### {title}")
    try:
        yield inner
    finally:
        outer.markdown("</div>", unsafe_allow_html=True)

def resolve_admin_key() -> str:
    """Read ADMIN_KEY from Streamlit secrets if available, otherwise fall back to env var."""
    try:
        return st.secrets["ADMIN_KEY"]
    except Exception:
        pass
    env_key = os.environ.get("ADMIN_KEY")
    if env_key:
        return env_key
    env_path = Path(".env")
    if env_path.exists():
        try:
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "ADMIN_KEY":
                    return value.strip().strip('"').strip("'")
        except Exception:
            pass
    return ""

def extract_numeric_user_id(value) -> int | None:
    """Return integer participant id if present in a string or number."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"\d+", str(value))
    if match:
        try:
            return int(match.group())
        except ValueError:
            return None
    return None

def load_last_user_id_from_logs() -> int:
    """Inspect study_logs to find the highest numeric user id."""
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

def ensure_participant_counter():
    """Initialize participant counter based on persisted logs."""
    if st.session_state.get("next_user_id") is None:
        st.session_state["next_user_id"] = load_last_user_id_from_logs() + 1

ensure_participant_counter()

ADMIN_KEY = resolve_admin_key()

def switch_page(new_page: str):
    """Switch pages and log time spent on the previous one."""
    now = pd.Timestamp.now()
    prev_page = st.session_state.get("current_page")
    entry_time = st.session_state.get("page_entry_time")
    if prev_page and entry_time is not None and new_page != prev_page:
        duration_seconds = float((now - entry_time).total_seconds())
        st.session_state["interaction_log"].append({
            "timestamp": now.isoformat(),
            "event": "page_duration",
            "page": prev_page,
            "duration_seconds": duration_seconds,
            "user_id": st.session_state.get("user_id"),
            "group": st.session_state.get("study_mode"),
        })
    if new_page != prev_page:
        st.session_state["current_page"] = new_page
        st.session_state["page_entry_time"] = now

# ----------------------------
# Session state initialization
# ----------------------------
if "study_mode" not in st.session_state:
    st.session_state["study_mode"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = ""
if "interaction_log" not in st.session_state:
    st.session_state["interaction_log"] = []
if "admin_mode" not in st.session_state:
    st.session_state["admin_mode"] = False
if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "participant"
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Assessment"
if "page_entry_time" not in st.session_state:
    st.session_state["page_entry_time"] = pd.Timestamp.now()
if "study_analysis_results" not in st.session_state:
    st.session_state["study_analysis_results"] = None
if "next_user_id" not in st.session_state:
    st.session_state["next_user_id"] = None
if "current_demographics" not in st.session_state:
    st.session_state["current_demographics"] = None
if "pre_info_submitted" not in st.session_state:
    st.session_state["pre_info_submitted"] = False
if "pre_info_answers" not in st.session_state:
    st.session_state["pre_info_answers"] = None

# ----------------------------
# Study mode selection
# ----------------------------
st.title("🧠 Burnout Assessment System")

if not st.session_state["admin_mode"]:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

def auto_assign_participant_if_needed():
    """Automatically assign participant id and study group (odd=G1, even=G2)."""
    if st.session_state["view_mode"] != "participant":
        return
    if st.session_state["user_id"]:
        return
    ensure_participant_counter()
    next_id = st.session_state["next_user_id"]
    st.session_state["next_user_id"] = next_id + 1
    user_id_str = str(next_id)
    group = "G2" if next_id % 2 == 0 else "G1"
    st.session_state["user_id"] = user_id_str
    st.session_state["study_mode"] = group
    st.session_state["interaction_log"].append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "event": "session_start",
        "user_id": user_id_str,
        "group": group,
        "assignment": "auto_counter"
    })
    st.rerun()

auto_assign_participant_if_needed()

with st.expander("🔐 Researcher console", expanded=False):
    if st.session_state["admin_mode"]:
        st.success("Researcher controls unlocked.")
        view_choice = st.radio(
            "Choose view",
            options=["participant", "admin"],
            index=0 if st.session_state["view_mode"] == "participant" else 1,
            format_func=lambda x: "Participant experience" if x == "participant" else "Research dashboard",
            horizontal=True,
        )
        if view_choice != st.session_state["view_mode"]:
            st.session_state["view_mode"] = view_choice
            st.rerun()
        if st.button("Lock console"):
            st.session_state["admin_mode"] = False
            st.session_state["view_mode"] = "participant"
            st.rerun()
    else:
        if not ADMIN_KEY:
            st.caption("Set ADMIN_KEY in Streamlit secrets or the environment to require a passphrase.")
        admin_key = st.text_input("Enter admin key", type="password")
        if st.button("Unlock console"):
            if not ADMIN_KEY or admin_key == ADMIN_KEY:
                st.session_state["admin_mode"] = True
                st.session_state["view_mode"] = "admin"
                st.success("Admin mode enabled. Sidebar and analytics will appear.")
                st.rerun()
            else:
                st.error("Incorrect key. Please try again.")

# Display current mode
# if st.session_state["view_mode"] == "participant":
#     mode_label = "Basic Assessment" if st.session_state["study_mode"] == "G1" else "Enhanced Assessment"
#     if st.session_state["admin_mode"]:
#         st.sidebar.success(f"Mode: {mode_label}")
#         st.sidebar.caption(f"Participant: {st.session_state['user_id'] or '—'}")
#     else:
#         participant_label = st.session_state["user_id"] or "—"
#         st.caption(f"**Mode:** {mode_label} | **Participant ID:** {participant_label}")
#         if st.session_state["user_id"]:
#             st.info(
#                 f"You are Participant #{participant_label}. "
#                 f"All odd numbers join Group 1, even numbers join Group 2. "
#                 f"Your assignment: **{mode_label}**."
#             )

# ----------------------------
# Pre-assessment info gate
# ----------------------------
if st.session_state["view_mode"] == "participant" and not st.session_state["pre_info_submitted"]:
    st.markdown("---")
    st.header("Quick Pre-Study Questions")
    st.caption("Before you start, please answer these to tailor the experience.")

    with st.form("pre_study_info"):
        field_of_study_work = st.text_input(
            "Field of study/work",
            placeholder="Ex: Computer science, nursing, finance, education"
        )
        ai_familiarity = st.select_slider(
            "Familiarity with AI systems (1–5)",
            options=[1, 2, 3, 4, 5],
            value=3
        )
        health_app_use = st.selectbox(
            "Frequency of health app/wearable use",
            ["Never", "Monthly", "Weekly", "Several times/week", "Daily"],
            index=2
        )
        pre_submit = st.form_submit_button("Continue", type="primary", use_container_width=True)

    if pre_submit:
        st.session_state["pre_info_answers"] = {
            "field_of_study_work": field_of_study_work,
            "ai_familiarity": ai_familiarity,
            "health_app_use": health_app_use,
        }
        st.session_state["pre_info_submitted"] = True
        st.session_state["interaction_log"].append({
            "timestamp": pd.Timestamp.now().isoformat(),
            "event": "pre_info_completed",
            "user_id": st.session_state["user_id"],
            "group": st.session_state["study_mode"],
            "pre_info": st.session_state["pre_info_answers"],
        })
        st.success("Thanks! Loading your assessment...")
        st.rerun()
    st.stop()

# ----------------------------
# Participant navigation
# ----------------------------
if st.session_state["view_mode"] == "participant":
    page_options = ["Assessment", "Results", "Survey"]
    current_page = st.session_state.get("current_page", page_options[0])
    if current_page not in page_options:
        current_page = page_options[0]
        st.session_state["current_page"] = current_page
    page_choice = st.radio(
        "Study navigation",
        page_options,
        index=page_options.index(current_page),
        horizontal=True,
    )
    if page_choice != current_page:
        switch_page(page_choice)

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
    
    if st.session_state.get("study_mode") != "G1":
        st.markdown("### 🎯 Your Burnout Factors")
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

def interpret_confidence(lower: float, upper: float) -> Dict[str, str]:
    """Convert numeric interval into plain-language cues."""
    width = upper - lower
    if width <= 0.10:
        level = "High confidence"
        explainer = (
            "We repeatedly land within a narrow ±5 percentage point band for people like you."
        )
    elif width <= 0.20:
        level = "Moderate confidence"
        explainer = (
            "Your score usually sits inside a medium-sized band; daily habits can swing it a bit."
        )
    else:
        level = "Needs more information"
        explainer = (
            "We do not see enough consistent data for similar profiles, so we show a wide safety band."
        )
    return {"level": level, "explanation": explainer, "width": width}

def explain_uncertainty_sources(aleatoric: float, epistemic: float) -> str:
    """Explain where uncertainty comes from without jargon."""
    if aleatoric >= epistemic + 0.02:
        return (
            "Most of the wiggle room comes from day-to-day lifestyle swings (sleep, breaks, caffeine)."
        )
    if epistemic >= aleatoric + 0.02:
        return (
            "The model has seen fewer people with a similar pattern, so it keeps a wider safety margin."
        )
    return (
        "Both everyday fluctuations and limited look-alike data contribute equally to this range."
    )

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

def compute_coverage_signal(width: float, ood_flag: bool) -> Dict[str, str]:
    """Summarize how well the model has seen similar people."""
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
    """Return a plain-language explanation for a driver contribution."""
    name = friendly_feature_name(feature)
    direction = "raises" if impact > 0 else "reduces"
    value = user_data.get(feature, "N/A")
    if isinstance(value, (int, float)):
        lo, hi = ranges.get(feature, (None, None))
        qualifier = ""
        if lo is not None and hi is not None:
            midpoint = (lo + hi) / 2
            if value >= hi:
                qualifier = "- that is among the highest values we usually see"
            elif value <= lo:
                qualifier = "- that is among the lowest values we usually see"
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

def format_display_value(value) -> str:
    """Format numeric or categorical values for UI labels."""
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)

def run_study_analysis_pipeline():
    """Execute analyze_study helpers and return structured outputs for the admin dashboard."""
    if study_analysis is None:
        return {"error": "analyze_study.py is not available in this workspace."}
    logs_df = study_analysis.load_study_data()
    if logs_df is None or logs_df.empty:
        return {"error": "No study log files found in study_logs/."}
    surveys = study_analysis.extract_survey_data(logs_df)
    if surveys is None or surveys.empty:
        return {"error": "No survey responses have been recorded yet."}
    warnings = []
    group_counts = surveys["group"].value_counts()
    if (group_counts < 5).any():
        warnings.append("Some study groups have fewer than 5 survey responses; interpret statistics carefully.")
    stats_df, surveys_with_metrics = study_analysis.compute_statistics(surveys)
    try:
        study_analysis.analyze_correlations(surveys_with_metrics)
    except Exception as exc:
        warnings.append(f"Correlation analysis failed: {exc}")
    figure_paths: List[Path] = []
    output_dir = getattr(study_analysis, "OUTPUT_DIR", Path("study_outputs"))
    try:
        study_analysis.create_visualizations(surveys_with_metrics)
        figure_paths = [
            path for name in ["results_boxplots.png", "results_barplot.png", "results_distributions.png"]
            if (path := output_dir / name).exists()
        ]
    except Exception as exc:
        warnings.append(f"Visualization generation failed: {exc}")
    export_paths: List[Path] = []
    try:
        study_analysis.export_results(surveys_with_metrics, stats_df)
        export_paths = [
            path for name in ["survey_responses.csv", "statistical_results.csv", "summary_statistics.csv"]
            if (path := output_dir / name).exists()
        ]
    except Exception as exc:
        warnings.append(f"Export step failed: {exc}")
    report_text = ""
    report_path = output_dir / "study_report.txt"
    try:
        study_analysis.generate_report(surveys_with_metrics, stats_df)
        if report_path.exists():
            report_text = report_path.read_text()
    except Exception as exc:
        warnings.append(f"Report generation failed: {exc}")
    qualitative_cols = [c for c in ["resp.q_open_most_useful", "resp.q_open_unclear", "resp.q_open_suggestions", "resp.q_open_uncertainty_impact"] if c in surveys_with_metrics.columns]
    qualitative = pd.DataFrame()
    if qualitative_cols:
        qualitative = surveys_with_metrics[["group"] + qualitative_cols].dropna(how="all", subset=qualitative_cols)
    return {
        "events": logs_df,
        "surveys": surveys_with_metrics,
        "stats": stats_df,
        "warnings": warnings,
        "figures": figure_paths,
        "exports": export_paths,
        "report_text": report_text,
        "report_path": report_path if report_path.exists() else None,
        "qualitative": qualitative,
    }
def save_interaction_log():
    """Save interaction log to JSON file."""
    log_dir = Path("study_logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{st.session_state['user_id']}_{st.session_state['study_mode']}_{timestamp}.json"
    
    with open(log_dir / filename, 'w') as f:
        json.dump(st.session_state["interaction_log"], f, indent=2)
    
    message = "✅ Session data saved"
    if st.session_state.get("admin_mode"):
        st.sidebar.success(message)
    else:
        st.success(message)

# ----------------------------
# Data loading
# ----------------------------
uploaded = None
if st.session_state["admin_mode"]:
    st.sidebar.header("📊 Data Management")
    uploaded = st.sidebar.file_uploader("Upload burnout dataset (CSV)", type=["csv"])

@st.cache_data(show_spinner=True)
def load_default_sample() -> pd.DataFrame | None:
    """Load default Burnout Dataset."""
    try:
        df = pd.read_csv("data/stress_risk_data.csv")
        return df
    except Exception:
        st.error("Could not load default dataset. Please upload a CSV file.")
        return None

if uploaded:
    df = pd.read_csv(uploaded)
    if st.session_state["admin_mode"]:
        st.sidebar.success("✅ Dataset loaded from upload.")
else:
    df = load_default_sample()
    if df is not None and st.session_state["admin_mode"]:
        st.sidebar.info("Using burnout dataset.")

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

if st.session_state["admin_mode"]:
    st.sidebar.header("🔬 Study Analytics")
    st.sidebar.metric("Rows in dataset", f"{len(df):,}")
    st.sidebar.metric("Features in play", len([c for c in NUM_COLS + CAT_COLS if c in df.columns]))
    if "target_binary" in df.columns:
        high_share = df["target_binary"].mean() * 100
        st.sidebar.metric("High-risk share", f"{high_share:.1f}%")
    dataset_csv = df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "⬇️ Download dataset snapshot",
        dataset_csv,
        file_name="stress_risk_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if st.session_state["interaction_log"]:
        log_df = pd.json_normalize(st.session_state["interaction_log"])
        summary = (
            log_df.query("event == 'prediction_requested'")
            .groupby("group")["event"]
            .count()
            .rename("predictions")
        )
        if not summary.empty:
            st.sidebar.caption("Predictions collected by group:")
            st.sidebar.dataframe(summary, height=150)
        logs_csv = log_df.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button(
            "⬇️ Download session log",
            logs_csv,
            file_name="interaction_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

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

if st.session_state["admin_mode"]:
    st.sidebar.success("✅ Model trained successfully")
else:
    st.caption("✅ Model ready for this session")

def render_admin_dashboard(df: pd.DataFrame, arts: Artifacts):
    """Dedicated admin-only dashboard."""
    st.title("🧪 Research Dashboard")
    st.caption("Unlocked researcher view. Use the console above to return to the participant experience.")
    
    total_rows = len(df)
    num_numeric = len([c for c in NUM_COLS if c in df.columns])
    num_cats = len([c for c in CAT_COLS if c in df.columns])
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows in dataset", f"{total_rows:,}")
    col2.metric("Numeric features", num_numeric)
    col3.metric("Categorical features", num_cats)
    
    st.markdown("### 📊 Outcome Distribution")
    if "stress_risk_score" in df.columns:
        fig_target = px.histogram(
            df,
            x="stress_risk_score",
            nbins=30,
            title="Stress risk score distribution",
            color_discrete_sequence=["#1f77b4"],
        )
    elif "target_binary" in df.columns:
        fig_target = px.histogram(
            df,
            x="target_binary",
            title="Binary risk distribution",
            color_discrete_sequence=["#1f77b4"],
        )
    else:
        fig_target = None
    if fig_target is not None:
        fig_target.update_layout(height=300, bargap=0.05)
        st.plotly_chart(fig_target, use_container_width=True)
    else:
        st.info("No stress_risk_score column available to visualize.")
    
    st.markdown("### 🧠 Model Feature Importance (avg. LightGBM gain)")
    try:
        importances = np.mean([m.booster_.feature_importance(importance_type="gain") for m in arts.models], axis=0)
    except Exception:
        importances = np.mean([m.feature_importances_ for m in arts.models], axis=0)
    imp_df = (
        pd.DataFrame({"Feature": arts.feature_cols, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(15)
    )
    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Blues")
    fig_imp.update_layout(height=400, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("### 🧾 Recent Participant Activity")
    logs = st.session_state.get("interaction_log", [])
    if logs:
        log_df = pd.json_normalize(logs)
        st.dataframe(log_df.tail(200), use_container_width=True, height=260)
        cohort_counts = (
            log_df.query("event == 'prediction_requested'")
            .groupby("group")
            .size()
            .reset_index(name="predictions")
        )
        if not cohort_counts.empty:
            st.markdown("#### Predictions collected per study group")
            st.dataframe(cohort_counts, use_container_width=True, height=140)
        logs_csv = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download interaction log (CSV)",
            logs_csv,
            file_name="interaction_log.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No participant interactions recorded yet in this session.")
    
    st.markdown("### 📥 Data Snapshot")
    st.dataframe(df.head(20), use_container_width=True)
    dataset_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download dataset snapshot (CSV)",
        dataset_csv,
        file_name="stress_risk_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("### 🟢 Pre-Study Info (this session)")
    pre_entries = [
        entry for entry in st.session_state.get("interaction_log", [])
        if entry.get("event") == "pre_info_completed" and isinstance(entry.get("pre_info"), dict)
    ]
    if pre_entries:
        pre_df = pd.DataFrame([e["pre_info"] | {"user_id": e.get("user_id"), "group": e.get("group")} for e in pre_entries])
        summary_cols = ["field_of_study_work", "ai_familiarity", "health_app_use"]
        col_a, col_b, col_c = st.columns(3)
        if "field_of_study_work" in pre_df.columns:
            counts = pre_df["field_of_study_work"].value_counts().rename_axis("Answer").reset_index(name="Count")
            col_a.markdown("**Field of study/work**")
            col_a.dataframe(counts, use_container_width=True, height=180)
        if "ai_familiarity" in pre_df.columns:
            counts = pre_df["ai_familiarity"].value_counts().sort_index().rename_axis("Familiarity (1–5)").reset_index(name="Count")
            col_b.markdown("**AI familiarity (1–5)**")
            col_b.dataframe(counts, use_container_width=True, height=180)
        if "health_app_use" in pre_df.columns:
            counts = pre_df["health_app_use"].value_counts().rename_axis("Frequency").reset_index(name="Count")
            col_c.markdown("**Health app/wearable use**")
            col_c.dataframe(counts, use_container_width=True, height=180)
        st.download_button(
            "⬇️ Download pre-study responses (CSV)",
            pre_df.to_csv(index=False).encode("utf-8"),
            file_name="pre_study_responses.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No pre-study info submitted in this session yet.")

    st.markdown("### 🧪 Study Log Analysis")
    if study_analysis is None:
        st.warning("analyze_study.py is missing, so automated analysis cannot run.")
    else:
        run_clicked = st.button("Run analyze_study pipeline", key="run_study_analysis")
        if run_clicked:
            with st.spinner("Running analyze_study helpers..."):
                st.session_state["study_analysis_results"] = run_study_analysis_pipeline()
        analysis = st.session_state.get("study_analysis_results")
        if analysis:
            if analysis.get("error"):
                st.error(analysis["error"])
            else:
                if analysis.get("warnings"):
                    for warn in analysis["warnings"]:
                        st.warning(warn)
                st.subheader("Survey Statistics")
                st.dataframe(analysis["stats"], use_container_width=True)
                st.download_button(
                    "⬇️ Download statistical summary (CSV)",
                    analysis["stats"].to_csv(index=False).encode("utf-8"),
                    file_name="statistical_results_dashboard.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_stats_dashboard",
                )
                st.subheader("Survey Responses")
                st.dataframe(analysis["surveys"], use_container_width=True, height=300)
                st.download_button(
                    "⬇️ Download survey responses (CSV)",
                    analysis["surveys"].to_csv(index=False).encode("utf-8"),
                    file_name="survey_responses_dashboard.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_surveys_dashboard",
                )
                if analysis.get("qualitative") is not None and not analysis["qualitative"].empty:
                    st.subheader("Participant Comments")
                    comments_df = analysis["qualitative"].copy()
                    comments_df = comments_df.rename(columns={"group": "Group", "comments": "Comment"})
                    st.caption(f"{len(comments_df)} comments collected. Scroll to explore or download the CSV below.")
                    st.dataframe(comments_df, use_container_width=True, height=220)
                    st.download_button(
                        "⬇️ Download comments (CSV)",
                        comments_df.to_csv(index=False).encode("utf-8"),
                        file_name="participant_comments.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_comments_dashboard",
                    )
                if analysis.get("figures"):
                    st.subheader("Generated Figures")
                    for path in analysis["figures"]:
                        st.image(str(path), caption=path.name, use_column_width=True)
                if analysis.get("exports"):
                    st.subheader("Download Generated Files")
                    for path in analysis["exports"]:
                        with open(path, "rb") as f:
                            st.download_button(
                                f"⬇️ {path.name}",
                                f.read(),
                                file_name=path.name,
                                mime="text/csv",
                                use_container_width=True,
                                key=f"export_{path.name}",
                            )
                report_text = analysis.get("report_text")
                if report_text:
                    st.subheader("Study Report")
                    st.text_area("Report preview", report_text, height=250)
                    report_path = analysis.get("report_path")
                    if report_path and report_path.exists():
                        with open(report_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download study report (txt)",
                                f.read(),
                                file_name=report_path.name,
                                mime="text/plain",
                                use_container_width=True,
                                key="download_report",
                            )

if st.session_state["admin_mode"] and st.session_state["view_mode"] == "admin":
    render_admin_dashboard(df, arts)
    st.stop()

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
submitted = False

if st.session_state["current_page"] == "Assessment":
    st.markdown("---")
    st.header("📝 Your Information")

    with st.form("user_form"):
        st.subheader("Demographics & Work")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age = st.number_input("Age", 18, 80, 30)
        with col2:
            gender_choices = arts.cat_categories.get("gender", ["male", "female", "other"])
            if not gender_choices:
                gender_choices = ["male", "female", "other"]
            gender = st.selectbox("Gender", gender_choices)
        with col3:
            occ_choices = arts.cat_categories.get("occupation", ["engineer", "nurse", "student"])
            if not occ_choices:
                occ_choices = ["engineer", "nurse", "student"]
            occ_choices = [opt for opt in occ_choices if str(opt).lower() != "professor"]
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
        
        submitted = st.form_submit_button("🔍 Assess Burnout", type="primary", use_container_width=True)
else:
    st.info("Switch to the Assessment page to adjust your inputs.")

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

    demographics = {
        "age": age,
        "gender": gender,
        "occupation": occupation,
    }
    pre_info = st.session_state.get("pre_info_answers") or {}
    demographics.update(pre_info)
    
    pred = predict_user(user_data)
    st.session_state["current_prediction"] = pred
    st.session_state["current_user"] = user_data
    st.session_state["current_demographics"] = demographics
    
    st.session_state["interaction_log"].append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "event": "prediction_requested",
        "user_id": st.session_state["user_id"],
        "group": st.session_state["study_mode"],
        "user_data": user_data,
        "prediction": pred,
        "demographics": demographics
    })
    switch_page("Results")
    st.rerun()

# ----------------------------
# Results display (Results page only)
# ----------------------------
if st.session_state.get("current_page") == "Results":
    st.markdown("---")
    st.header("📊 Burnout Assessment Results")
    
<<<<<<< HEAD
    pred = st.session_state["current_prediction"]
    risk = pred["risk"]
    
    # G1: Basic display
    if st.session_state["study_mode"] == "G1":
        with result_card("Your Burnout Score", tag="Assessment Result") as card:
=======
    if "current_prediction" not in st.session_state:
        st.info("Submit the assessment on the Assessment page to view results.")
    else:
        pred = st.session_state["current_prediction"]
        risk = pred["risk"]
        
        # G1: Basic display
        if st.session_state["study_mode"] == "G1":
            st.markdown("### Your Burnout Score")
            
>>>>>>> c9289ae4efcedf1d4f9b6fe3310b818ddb7efadf
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk * 100,
                title = {'text': "Burnout Level (%)"},
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
            card.plotly_chart(fig, use_container_width=True)
            
            if risk < 0.33:
<<<<<<< HEAD
                card.success("✅ Low Burnout")
            elif risk < 0.66:
                card.warning("⚠️ Moderate Burnout")
            else:
                card.error("🚨 High Burnout")
    
    # G2: Enhanced display
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            lo = pred["uncertainty"]["lower"]
            hi = pred["uncertainty"]["upper"]
            confidence = interpret_confidence(lo, hi)
            coverage = compute_coverage_signal(hi - lo, pred["ood_flag"])
            aleatoric_std = pred["uncertainty"].get("aleatoric_std", 0.0)
            epistemic_std = pred["uncertainty"].get("epistemic_std", 0.0)

            with result_card("Burnout with Uncertainty", tag="Assessment Result") as card:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk * 100,
                    title = {'text': "Burnout Level (%)"},
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
                card.plotly_chart(fig, use_container_width=True)

            with result_card("Understanding this result", tag="Interpretation") as card:
                card.markdown("#### How to read this")
                snapshot_cols = card.columns([1.2, 1, 1])
                snapshot_cols[0].metric("Confidence level", confidence["level"])
                snapshot_cols[1].metric("Lower bound", f"{lo*100:.1f}%")
                snapshot_cols[2].metric("Upper bound", f"{hi*100:.1f}%")
                card.write(confidence["explanation"])
                card.info(
                    f"Out of 100 similar people, about 90 are expected to land between {lo*100:.0f}% and {hi*100:.0f}% burnout."
                )
                card.markdown("#### Why do we show a range?")
                card.write(explain_uncertainty_sources(aleatoric_std, epistemic_std))
                
                card.markdown("#### Two types of uncertainty")
                total_unc = aleatoric_std + epistemic_std
                if total_unc <= 1e-4:
                    card.info(
                        "Both natural variation and AI knowledge gaps are essentially zero right now, so the model is very "
                        "confident and the bar chart would be empty. If the range widens later, the bars will show which "
                        "source is responsible."
                    )
                else:
                    fig_unc = go.Figure(data=[
                        go.Bar(
                            name="Natural variation",
                            y=["Sources"],
                            x=[aleatoric_std * 100],
                            orientation="h",
                            marker_color="#58c4dd",
                            text=f"{aleatoric_std * 100:.1f}%",
                            textposition="inside"
                        ),
                        go.Bar(
                            name="AI knowledge gaps",
                            y=["Sources"],
                            x=[epistemic_std * 100],
                            orientation="h",
                            marker_color="#ffa07a",
                            text=f"{epistemic_std * 100:.1f}%",
                            textposition="inside"
                        ),
                    ])
                    fig_unc.update_layout(
                        barmode="stack",
                        height=220,
                        xaxis_title="Share of total uncertainty (%)",
                        yaxis=dict(showticklabels=False),
                        margin=dict(l=10, r=10, t=10, b=10),
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
                    )
                    card.plotly_chart(fig_unc, use_container_width=True)
                    card.caption(
                        "Longer bars mean that source takes up more of the uncertainty band around your score."
                    )
                card.markdown("#### What do these mean?")
                meaning_cols = card.columns(2)
                meaning_cols[0].markdown(
                    """
                    **🌊 Natural variation** (aleatoric)  
                    • Stress swings from day to day  
                    • Everyone keeps a buffer because no habits are perfectly steady  
                    • This portion cannot be eliminated—it is daily-life noise
                    """
                )
                meaning_cols[1].markdown(
                    """
                    **🧩 AI knowledge gaps** (epistemic)  
                    • The AI has seen fewer people with your exact pattern  
                    • More experience (additional similar data) shrinks this slice  
                    • Think new doctor vs. specialist—the specialist hedges less
                    """
                )
                
                card.markdown("#### Uncertainty metrics")
                unc_width = hi - lo
                metric_cols = card.columns(3)
                metric_cols[0].metric("Lower bound", f"{lo*100:.1f}%")
                metric_cols[1].metric("Upper bound", f"{hi*100:.1f}%")
                metric_cols[2].metric("Range (±)", f"{unc_width*50:.1f}%")
                
                if coverage["tone"] == "success":
                    card.success(f"{coverage['label']}: {coverage['message']}")
                elif coverage["tone"] == "warning":
                    card.warning(f"{coverage['label']}: {coverage['message']}")
                else:
                    card.info(f"{coverage['label']}: {coverage['message']}")
        
        with col2:
            with result_card("Key Stress Drivers", tag="Insights") as card:
                card.markdown("Factors with the most impact on your risk score:")
                
                drivers = pred["drivers"]
                driver_df = pd.DataFrame(drivers, columns=["Factor", "Impact"])
                driver_df["Direction"] = driver_df["Impact"].apply(
                    lambda x: "Increases ↑" if x > 0 else "Decreases ↓"
                )
=======
                st.success("✅ Low Burnout")
            elif risk < 0.66:
                st.warning("⚠️ Moderate Burnout")
            else:
                st.error("🚨 High Burnout")
        
        # G2: Enhanced display
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Burnout with Uncertainty")
                
                lo = pred["uncertainty"]["lower"]
                hi = pred["uncertainty"]["upper"]
                confidence = interpret_confidence(lo, hi)
                coverage = compute_coverage_signal(hi - lo, pred["ood_flag"])
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk * 100,
                    title = {'text': "Burnout Level (%)"},
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
                
                st.markdown("#### How to read this")
                st.markdown(f"**Confidence level:** {confidence['level']}")
                st.write(confidence["explanation"])
                st.caption(
                    f"In everyday terms: out of 100 similar people, about 90 are expected to land between "
                    f"{lo*100:.0f}% and {hi*100:.0f}% burnout."
                )
                aleatoric_std = pred["uncertainty"].get("aleatoric_std", 0.0)
                epistemic_std = pred["uncertainty"].get("epistemic_std", 0.0)
                st.markdown("#### Why do we show a range?")
                st.markdown(
                    f"{explain_uncertainty_sources(aleatoric_std, epistemic_std)}"
                )
                
                st.markdown("#### Two Types of Uncertainty")
                total_unc = aleatoric_std + epistemic_std
                if total_unc <= 1e-4:
                    st.info(
                        "Both natural variation and AI knowledge gaps are essentially zero right now, so the model is very "
                        "confident and the bar chart would be empty. If the range widens later, the bars will show which "
                        "source is responsible."
                    )
                else:
                    fig_unc = go.Figure(data=[
                        go.Bar(
                            name="Natural variation",
                            y=["Sources"],
                            x=[aleatoric_std * 100],
                            orientation="h",
                            marker_color="#58c4dd",
                            text=f"{aleatoric_std * 100:.1f}%",
                            textposition="inside"
                        ),
                        go.Bar(
                            name="AI knowledge gaps",
                            y=["Sources"],
                            x=[epistemic_std * 100],
                            orientation="h",
                            marker_color="#ffa07a",
                            text=f"{epistemic_std * 100:.1f}%",
                            textposition="inside"
                        ),
                    ])
                    fig_unc.update_layout(
                        barmode="stack",
                        height=220,
                        xaxis_title="Share of total uncertainty (%)",
                        yaxis=dict(showticklabels=False),
                        margin=dict(l=10, r=10, t=10, b=10),
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
                    )
                    st.plotly_chart(fig_unc, use_container_width=True)
                    st.caption(
                        "Longer bars mean that source takes up more of the uncertainty band around your score."
                    )
                with st.expander("💡 What do these mean?", expanded=False):
                    st.markdown("""
                    **🌊 Natural variation** (aleatoric uncertainty)  
                    • Stress naturally swings from day to day, just like your heart rate or weight  
                    • The model always keeps this buffer because no one has perfectly steady habits  
                    • This part cannot be eliminated- it is the noise of daily life  

                    **🧩 AI knowledge gaps** (epistemic uncertainty)  
                    • The AI has seen fewer people with your exact pattern, so it hedges its bets  
                    • Think of a new doctor versus a specialist-experience narrows this portion  
                    • Collecting more similar data points would shrink this slice
                    """)
                
                st.markdown("#### Uncertainty Metrics")
                unc_width = hi - lo
                c1, c2, c3 = st.columns(3)
                c1.metric("Lower Bound", f"{lo*100:.1f}%")
                c2.metric("Upper Bound", f"{hi*100:.1f}%")
                c3.metric("Range", f"±{unc_width*50:.1f}%")
                st.caption(
                    "Lower bound: the most cautious estimate. Upper bound: the highest likely score. "
                    "Range: how much padding we keep on each side of the best estimate."
                )
                
                if coverage["tone"] == "success":
                    st.success(f"{coverage['label']}: {coverage['message']}")
                elif coverage["tone"] == "warning":
                    st.warning(f"{coverage['label']}: {coverage['message']}")
                else:
                    st.info(f"{coverage['label']}: {coverage['message']}")
        
            with col2:
                st.markdown("### Key Stress Drivers")
                st.markdown("Factors with the most impact on your risk score:")
                
                drivers = pred["drivers"]
                driver_df = pd.DataFrame(drivers, columns=["Factor", "Impact"])
                driver_df["Direction"] = driver_df["Impact"].apply(
                    lambda x: "Increases ↑" if x > 0 else "Decreases ↓"
                )
>>>>>>> c9289ae4efcedf1d4f9b6fe3310b818ddb7efadf
                
                fig_shap = px.bar(
                    driver_df, y="Factor", x="Impact", orientation='h',
                    color="Direction",
                    color_discrete_map={"Increases ↑": "#ff6b6b", "Decreases ↓": "#51cf66"}
                )
                fig_shap.update_layout(height=350, yaxis={'categoryorder':'total ascending'})
<<<<<<< HEAD
                card.plotly_chart(fig_shap, use_container_width=True)
                
                card.markdown("#### Plain-English Reason Codes")
                for feat, impact in drivers:
                    arrow = "⚠️" if impact > 0 else "✅"
                    card.write(
                        f"{arrow} **{friendly_feature_name(feat)}** ({impact:+.2f}) - "
                        f"{describe_driver(feat, impact, st.session_state['current_user'], arts.ranges)}."
=======
                st.plotly_chart(fig_shap, use_container_width=True)
                
                st.markdown("#### Plain-English Reason Codes")
                for feat, impact in drivers:
                    arrow = "⚠️" if impact > 0 else "✅"
                    st.write(
                        f"{arrow} **{friendly_feature_name(feat)}** ({impact:+.2f}) - "
                        f"{describe_driver(feat, impact, st.session_state['current_user'], arts.ranges)}."
                    )
        
        st.markdown("### 🔁 What-if Sandbox")
        with st.expander("Try small changes and see how risk shifts:", expanded=False):
            baseline_user = st.session_state.get("current_user")
            if not baseline_user:
                st.caption("Submit the form above to unlock scenario testing.")
            else:
                adjustable = []
                for feat, _ in pred["drivers"]:
                    if feat in arts.feature_cols and feat not in adjustable:
                        adjustable.append(feat)
                if not adjustable:
                    st.caption("No adjustable factors available from the current explanation.")
                else:
                    feature_choice = st.selectbox(
                        "Pick a factor to tweak",
                        adjustable,
                        format_func=friendly_feature_name,
                        key="what_if_feature"
>>>>>>> c9289ae4efcedf1d4f9b6fe3310b818ddb7efadf
                    )
        
        with result_card("🔁 What-if Sandbox", tag="Scenario Lab") as card:
            with card.expander("Try small changes and see how risk shifts:", expanded=False):
                baseline_user = st.session_state.get("current_user")
                if not baseline_user:
                    st.caption("Submit the form above to unlock scenario testing.")
                else:
                    adjustable = []
                    for feat, _ in pred["drivers"]:
                        if feat in arts.feature_cols and feat not in adjustable:
                            adjustable.append(feat)
                    if not adjustable:
                        st.caption("No adjustable factors available from the current explanation.")
                    else:
                        feature_choice = st.selectbox(
                            "Pick a factor to tweak",
                            adjustable,
                            format_func=friendly_feature_name,
                            key="what_if_feature"
                        )
<<<<<<< HEAD
                        new_value = baseline_user.get(feature_choice)
                        control_key = f"what_if_value_{feature_choice}"
                        if feature_choice in NUM_COLS:
                            lo, hi = arts.ranges.get(feature_choice, (None, None))
                            if lo is None or hi is None:
                                current_val = float(baseline_user.get(feature_choice, 0))
                                lo = current_val - 5
                                hi = current_val + 5
                            if lo == hi:
                                hi = lo + 1.0
                            raw_val = baseline_user.get(feature_choice)
                            if raw_val is None:
                                base_val = (lo + hi) / 2
                            else:
                                base_val = float(raw_val)
                            base_val = min(max(base_val, lo), hi)
                            is_continuous = abs(base_val - round(base_val)) > 0.01
                            step = 0.5 if is_continuous else 1.0
                            new_value = st.slider(
                                f"Set a new {friendly_feature_name(feature_choice)}",
                                float(lo),
                                float(hi),
                                float(base_val),
                                step=step,
                                key=control_key
                            )
                        elif feature_choice in CAT_COLS:
                            choices = arts.cat_categories.get(feature_choice, [])
                            if not choices:
                                choices = sorted({str(baseline_user.get(feature_choice, ""))})
                            current_choice = str(baseline_user.get(feature_choice, choices[0] if choices else ""))
                            idx = choices.index(current_choice) if current_choice in choices else 0
                            new_value = st.selectbox(
                                f"Set a new {friendly_feature_name(feature_choice)}",
                                choices,
                                index=idx if choices else 0,
                                key=control_key
                            )
                        run = st.button("Run scenario", key="what_if_run")
                        if run:
                            scenario = dict(baseline_user)
                            scenario[feature_choice] = new_value
                            scenario_pred = predict_user(scenario)
                            st.session_state["what_if_result"] = {
                                "feature": feature_choice,
                                "value": new_value,
                                "prediction": scenario_pred,
                                "baseline": risk
                            }
                        if "what_if_result" in st.session_state:
                            result = st.session_state["what_if_result"]
                            delta = (result["prediction"]["risk"] - result["baseline"]) * 100
                            value_display = format_display_value(result["value"])
                            st.metric(
                                label=f"New risk if {friendly_feature_name(result['feature'])} = {value_display}",
                                value=f"{result['prediction']['risk']*100:.1f}%",
                                delta=f"{delta:+.1f} pts vs now"
                            )
    
    # Stress management plan
    st.markdown("---")
    st.header("📅 Your Personalized Stress Management Plan")
    
    if st.button("Generate My Plan", type="primary", use_container_width=True):
        plan = generate_stress_management_plan(st.session_state["current_user"], pred)
        st.session_state["stress_plan"] = plan
=======
>>>>>>> c9289ae4efcedf1d4f9b6fe3310b818ddb7efadf
        
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
            st.caption("When you are ready, move to the Survey page to share your feedback.")

# ----------------------------
# Questionnaire (Survey page only)
# ----------------------------
if st.session_state.get("current_page") == "Survey":
    st.markdown("---")
    st.header("📋 Burnout Assessment - User Study Questionnaire")
    st.caption("Please answer after reviewing your results/plan. Use 1 = strongly disagree to 5 = strongly agree unless noted.")

    if "current_prediction" not in st.session_state:
        st.info("Complete the assessment and view your results before filling out the survey.")
    else:
        driver_options = [
            "Weekly work hours",
            "Workload rating",
            "Stress events last week",
            "Sleep duration",
            "Phone unlocks per day",
            "Screen time",
            "I'm not sure",
        ]

        demo_defaults = st.session_state.get("current_demographics") or {}

        with st.form("study_questionnaire"):
            likert_scale_5 = [1, 2, 3, 4, 5]
            likert_scale_7 = [1, 2, 3, 4, 5, 6, 7]
            responses: Dict[str, object] = {}

            def likert(label: str, key: str, scale: list[int], default: int) -> int:
                return st.select_slider(label, options=scale, value=default, key=key)

            st.subheader("1 - Demographics & Background")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                default_age = demo_defaults.get("age", 30)
                try:
                    default_age = int(default_age)
                except Exception:
                    default_age = 30
                responses["q_demo_age"] = st.number_input("Age", 18, 90, default_age, key="q_demo_age")
            with col_b:
                gender_opts = arts.cat_categories.get("gender", ["male", "female", "other"])
                if not gender_opts:
                    gender_opts = ["male", "female", "other"]
                gender_default = str(demo_defaults.get("gender", "")).lower()
                gender_index = gender_opts.index(gender_default) if gender_default in gender_opts else 0
                responses["q_demo_gender"] = st.selectbox("Gender", gender_opts, index=gender_index, key="q_demo_gender")
            with col_c:
                responses["q_demo_occupation"] = st.text_input("Occupation", value=str(demo_defaults.get("occupation", "")), key="q_demo_occupation")
            responses["q_demo_education"] = st.selectbox(
                "Highest education level",
                ["High school", "Associate degree", "Bachelor's degree", "Master's degree", "Doctorate", "Other"],
                key="q_demo_education"
            )
            responses["q_demo_field_of_study_work"] = st.text_input(
                "Field of study/work",
                value=str(demo_defaults.get("field_of_study_work", "")),
                key="q_demo_field_of_study_work"
            )

            st.subheader("2 - Health & Stress Context")
            responses["q_health_stress_level"] = likert(
                "Self-rated current stress/mental health level (1 = very low, 5 = very high)",
                "q_health_stress_level",
                likert_scale_5,
                3
            )
            responses["q_health_prior_tool_use"] = st.radio(
                "Prior use of stress-assessment or mental-health tools",
                ["No", "Yes"],
                key="q_health_prior_tool_use"
            )

            st.subheader("3 - AI Literacy & Attitudes")
            responses["q_ai_knowledge"] = likert(
                "Self-rated AI knowledge (1 = none, 5 = advanced)",
                "q_ai_knowledge",
                likert_scale_5,
                3
            )
            responses["q_ai_tool_frequency"] = st.selectbox(
                "Frequency of AI tool use",
                ["Never", "Occasionally", "Weekly", "Daily"],
                key="q_ai_tool_frequency"
            )
            st.markdown("**GAAIS-10 (Validated attitude items)**")
            responses["q_gaais_interest"] = likert("I am interested in using artificial intelligence systems in my daily life.", "q_gaais_interest", likert_scale_5, 3)
            responses["q_gaais_impact"] = likert("Artificial Intelligence can have positive impacts on people's wellbeing.", "q_gaais_impact", likert_scale_5, 3)
            responses["q_gaais_excitement"] = likert("Artificial Intelligence is exciting.", "q_gaais_excitement", likert_scale_5, 3)
            responses["q_gaais_benefit"] = likert("Much of society will benefit from a future full of Artificial Intelligence.", "q_gaais_benefit", likert_scale_5, 3)
            responses["q_gaais_employment"] = likert("I would like to use artificial intelligence in my own job.", "q_gaais_employment", likert_scale_5, 3)
            responses["q_gaais_sinister"] = likert("I find artificial intelligence sinister.", "q_gaais_sinister", likert_scale_5, 3)
            responses["q_gaais_control"] = likert("Artificial Intelligence might take control of people.", "q_gaais_control", likert_scale_5, 3)
            responses["q_gaais_danger"] = likert("I think artificial intelligence is dangerous.", "q_gaais_danger", likert_scale_5, 3)
            responses["q_gaais_discomfort"] = likert("I shiver with discomfort when I think about future uses of artificial intelligence.", "q_gaais_discomfort", likert_scale_5, 3)
            responses["q_gaais_suffering"] = likert("People like me will suffer if artificial intelligence is used more and more.", "q_gaais_suffering", likert_scale_5, 3)

            st.markdown("**Health App Experience**")
            responses["q_health_app_frequency"] = st.selectbox(
                "Frequency of health app or wearable use",
                ["Never", "Monthly", "Weekly", "Daily"],
                key="q_health_app_frequency"
            )

            st.subheader("Core Survey Items (TAM)")
            st.markdown("**Perceived Usefulness**")
            responses["q_tam_pu_useful"] = likert("Using this system would be useful for managing my stress/mental health.", "q_tam_pu_useful", likert_scale_7, 4)
            responses["q_tam_pu_effective"] = likert("Using this system would improve how effectively I manage my stress/mental health.", "q_tam_pu_effective", likert_scale_7, 4)
            responses["q_tam_pu_productive"] = likert("Using this system would make me more productive in taking care of my mental health.", "q_tam_pu_productive", likert_scale_7, 4)
            responses["q_tam_pu_easier"] = likert("Using this system would make it easier to manage my stress/mental health.", "q_tam_pu_easier", likert_scale_7, 4)
            responses["q_tam_pu_understand"] = likert("Using this system would enhance my ability to understand my stress level.", "q_tam_pu_understand", likert_scale_7, 4)
            responses["q_tam_pu_value"] = likert("Overall, I find this system valuable for my mental health decisions.", "q_tam_pu_value", likert_scale_7, 4)

            st.markdown("**Perceived Ease of Use**")
            responses["q_tam_peou_learn"] = likert("Learning to use this system would be easy for me.", "q_tam_peou_learn", likert_scale_7, 4)
            responses["q_tam_peou_clear"] = likert("My interaction with the system would be clear and understandable.", "q_tam_peou_clear", likert_scale_7, 4)
            responses["q_tam_peou_skillful"] = likert("I would find it easy to become skillful at using the system.", "q_tam_peou_skillful", likert_scale_7, 4)
            responses["q_tam_peou_flexible"] = likert("I would find the system flexible to interact with.", "q_tam_peou_flexible", likert_scale_7, 4)
            responses["q_tam_peou_do_what"] = likert("I would find it easy to get the system to do what I want it to do.", "q_tam_peou_do_what", likert_scale_7, 4)
            responses["q_tam_peou_easy"] = likert("Overall, I would find the system easy to use.", "q_tam_peou_easy", likert_scale_7, 4)
            responses["q_attention_check_feature"] = st.selectbox(
                "Attention Check: Which factor increased your stress risk the most?",
                driver_options,
                key="q_attention_check_feature"
            )
            responses["q_system_preference"] = st.radio(
                "Which version of the system do you prefer: Basic or Advanced?",
                ["Basic", "Advanced", "No preference"],
                index=1 if st.session_state["study_mode"] == "G2" else 0,
                key="q_system_preference"
            )

            st.subheader("Open-Ended Questions")
            responses["q_open_most_useful"] = st.text_area("What part of the assessment was most useful?", key="q_open_most_useful")
            responses["q_open_unclear"] = st.text_area("What part was unclear?", key="q_open_unclear")
            responses["q_open_suggestions"] = st.text_area("Do you have suggestions for improvement?", key="q_open_suggestions")

            if st.session_state["study_mode"] == "G2":
                st.subheader("Group 2 Only – Uncertainty & Explanations")
                st.markdown("**Explanation Satisfaction Scale (ESS)**")
                responses["q_ess_understand"] = likert("From the explanation, I understand how the system works.", "q_ess_understand", likert_scale_5, 3)
                responses["q_ess_satisfying"] = likert("This explanation of how the system works is satisfying.", "q_ess_satisfying", likert_scale_5, 3)
                responses["q_ess_detail"] = likert("This explanation of how the system works has sufficient detail.", "q_ess_detail", likert_scale_5, 3)
                responses["q_ess_complete"] = likert("This explanation seems complete.", "q_ess_complete", likert_scale_5, 3)
                responses["q_ess_accurate"] = likert("This explanation shows me how accurate the system is.", "q_ess_accurate", likert_scale_5, 3)
                responses["q_ess_reliable"] = likert("This explanation shows me how reliable the system is.", "q_ess_reliable", likert_scale_5, 3)
                responses["q_ess_how_to_use"] = likert("This explanation tells me how to use the system.", "q_ess_how_to_use", likert_scale_5, 3)
                responses["q_ess_useful_goals"] = likert("This explanation is useful to my goals.", "q_ess_useful_goals", likert_scale_5, 3)
                responses["q_ess_trust_calibration"] = likert("This explanation helps me know when I should trust and not trust the system.", "q_ess_trust_calibration", likert_scale_5, 3)

                st.markdown("**Uncertainty Visualization**")
                responses["q_uncertainty_understand_confidence"] = likert(
                    "The uncertainty visualization helped me understand the system’s confidence.",
                    "q_uncertainty_understand_confidence",
                    likert_scale_5,
                    3
                )
                responses["q_uncertainty_transparent"] = likert(
                    "Showing uncertainty made the prediction feel more transparent.",
                    "q_uncertainty_transparent",
                    likert_scale_5,
                    3
                )
                responses["q_open_uncertainty_impact"] = st.text_area(
                    "Did the uncertainty or explanations change how you interpreted the results?",
                    key="q_open_uncertainty_impact"
                )

            survey_submit = st.form_submit_button("Submit Questionnaire", use_container_width=True)

        if survey_submit:
            st.session_state["interaction_log"].append({
                "timestamp": pd.Timestamp.now().isoformat(),
                "event": "questionnaire_completed",
                "user_id": st.session_state["user_id"],
                "group": st.session_state["study_mode"],
                "responses": responses,
                "demographics": st.session_state.get("current_demographics"),
            })

            st.success("✅ Thank you for completing the study questionnaire!")
            save_interaction_log()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>This is a research tool for educational purposes.</p>
    <p>Your data is anonymized and used only for research.</p>
</div>
""", unsafe_allow_html=True)
