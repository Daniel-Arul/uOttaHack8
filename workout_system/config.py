"""Configuration settings for the workout system"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# API Configuration
YELLOWCAKE_API_KEY = os.getenv("YELLOWCAKE_API_KEY", "")
YELLOWCAKE_API_URL = "https://api.yellowcake.dev/v1/extract-stream"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEBUG_AI = os.getenv("DEBUG_AI", "").lower() == "true"

# Session Configuration
AI_CHECK_INTERVAL = 30  # Check AI every N frames
MIN_API_CALL_DELAY = 2.0  # Minimum delay between API calls (seconds)
