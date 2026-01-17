"""Main entry point for the workout system"""

from .config import GEMINI_API_KEY
from .session import run_interactive_stretch_session


def main():
    """Run the interactive workout session"""
    goal = "Flexibility"
    session_length_min = 5
    session_length_sec = session_length_min * 60

    if GEMINI_API_KEY:
        print("✓ AI Pose Classification ENABLED (using Google Gemini)")
    else:
        print("ℹ AI Pose Classification DISABLED (using rule-based verification)")
        print("  To enable: export GEMINI_API_KEY=\"your-api-key-here\"\n")

    website_url = None

    run_interactive_stretch_session(goal, session_length_sec, website_url=website_url)


if __name__ == "__main__":
    main()
