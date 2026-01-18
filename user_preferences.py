"""User preferences management"""

import json
import os
from pathlib import Path

PREFS_FILE = Path.home() / ".posture_monitor_prefs.json"

DEFAULT_PREFS = {
    "first_run": True,
    "selected_habits": [],
    "strictness_level": "medium",  # "strict", "medium", "relaxed"
}

STRICTNESS_TIMERS = {
    "strict": 10,      # 10 seconds
    "medium": 30,      # 30 seconds (default)
    "relaxed": 60,     # 60 seconds
}

HABIT_GOALS = {
    "Back Pain": "Back Pain",
    "Posture": "Posture",
    "Flexibility": "Flexibility",
    "Desk Breaks": "Desk Break",
    "Mobility for Sports": "Flexibility",
}


def load_preferences():
    """Load user preferences from file"""
    if PREFS_FILE.exists():
        try:
            with open(PREFS_FILE, "r") as f:
                prefs = json.load(f)
                return {**DEFAULT_PREFS, **prefs}
        except Exception:
            return DEFAULT_PREFS.copy()
    return DEFAULT_PREFS.copy()


def save_preferences(prefs):
    """Save user preferences to file"""
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(prefs, f, indent=2)
    except Exception as e:
        print(f"Error saving preferences: {e}")


def is_first_run():
    """Check if this is the first run"""
    prefs = load_preferences()
    return prefs.get("first_run", True)


def mark_first_run_complete(habits, strictness):
    """Mark first run as complete and save habits/strictness"""
    prefs = load_preferences()
    prefs["first_run"] = False
    prefs["selected_habits"] = habits
    prefs["strictness_level"] = strictness
    save_preferences(prefs)


def get_timer_duration():
    """Get the timer duration based on strictness level"""
    prefs = load_preferences()
    strictness = prefs.get("strictness_level", "medium")
    return STRICTNESS_TIMERS.get(strictness, 30)


def get_selected_habits():
    """Get selected habits"""
    prefs = load_preferences()
    return prefs.get("selected_habits", [])


def get_selected_goals():
    """Get goals based on selected habits"""
    habits = get_selected_habits()
    goals = set()
    for habit in habits:
        goal = HABIT_GOALS.get(habit)
        if goal:
            goals.add(goal)
    return list(goals) if goals else ["Back Pain"]