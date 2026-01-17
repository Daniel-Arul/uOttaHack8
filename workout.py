import random
import cv2
import numpy as np
import mediapipe as mp
import os
import time
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# Configuration
YELLOWCAKE_API_KEY = os.getenv("YELLOWCAKE_API_KEY", "")  # Set via environment variable
YELLOWCAKE_API_URL = "https://api.yellowcake.dev/v1/extract-stream"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # For Google Gemini AI pose classification
DEBUG_AI = os.getenv("DEBUG_AI", "").lower() == "true"  # Set DEBUG_AI=true to see AI calls

# --- Step 1: Stretch database ---
stretches = [
    {
        "name": "Cat-Cow",
        "target": "Back",
        "duration": 60,
        "instructions": "Get on hands and knees. Alternate between:\n  COW: Drop hips, lift head and chest up, arch your back\n  CAT: Hunch shoulders up, drop head, round your back\n  Repeat smoothly, syncing with breathing",
    },
    {
        "name": "Child's Pose",
        "target": "Back",
        "duration": 60,
        "instructions": "Kneel, then sit hips back to heels with forehead touching ground.\n  Arms can be extended forward or relaxed by your sides.\n  Keep your back rounded and shoulders relaxed.",
    },
    {
        "name": "Spinal Twist",
        "target": "Back",
        "duration": 60,
        "instructions": "Sit with legs extended. Bend one knee and cross it over other leg.\n  Twist torso toward the bent knee, hugging it to your chest.\n  Keep your back straight and twist from the spine.\n  Alternate sides.",
    },
    {
        "name": "Chest Opener",
        "target": "Chest/Shoulders",
        "duration": 60,
        "instructions": "Stand upright with feet shoulder-width apart.\n  Pull both arms back behind you, keeping elbows bent.\n  Squeeze shoulder blades together, open your chest.\n  Keep shoulders level and arms extended back.",
    },
    {
        "name": "Shoulder Rolls",
        "target": "Shoulders",
        "reps": 20,
        "instructions": "Stand upright. Raise shoulders up toward ears, then roll back.\n  Roll in smooth circles, then reverse direction.\n  Keep arms relaxed at sides.\n  Complete 20 rolls total.",
    },
    {
        "name": "Neck Side Stretch",
        "target": "Neck/Shoulders",
        "duration": 15,
        "instructions": "Sit or stand upright. Tilt head toward one shoulder.\n  Gently bring ear closer to shoulder (don't force it).\n  Alternate sides, one side at a time.",
    },
    {
        "name": "Hamstring Stretch",
        "target": "Legs",
        "duration": 60,
        "instructions": "Stand on one leg. Lift the other leg and hold it with hands.\n  Keep leg straight and bend forward at hips slightly.\n  Feel the stretch in the back of your leg.\n  Alternate legs.",
    },
    {
        "name": "Dynamic Lunges",
        "target": "Legs/Hips",
        "reps": 16,
        "instructions": "Step forward with one leg, bending both knees to 90 degrees.\n  Back knee should almost touch ground, front knee over ankle.\n  Push through front heel to return to start.\n  Alternate legs in a walking motion.\n  Complete 16 lunges (8 per leg).",
    },
    {
        "name": "Shoulder Shrugs",
        "target": "Shoulders",
        "reps": 15,
        "instructions": "Stand with arms at sides. Raise shoulders up toward ears.\n  Hold briefly, then release back down.\n  Repeat in a controlled, rhythmic motion.\n  Complete 15 shrugs.",
    },
]

# --- Step 2: Goal to stretches mapping ---
goal_to_stretches = {
    "Back Pain": ["Cat-Cow", "Spinal Twist", "Child's Pose"],
    "Posture": ["Chest Opener", "Shoulder Rolls"],
    "Flexibility": ["Hamstring Stretch", "Dynamic Lunges"],
    "Desk Break": ["Neck Side Stretch", "Shoulder Shrugs"],
}


def extract_pose_data(results):
    """Extract pose landmarks from MediaPipe results"""
    if not results.pose_landmarks:
        return None
    
    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    return np.array(landmarks)


def get_landmark(landmarks, idx):
    """Get a specific landmark (x, y, z) from the landmarks array"""
    if landmarks is None or len(landmarks) < (idx + 1) * 3:
        return None
    return landmarks[idx * 3 : (idx + 1) * 3]


def distance_between_landmarks(landmarks, idx1, idx2):
    """Calculate distance between two landmarks"""
    lm1 = get_landmark(landmarks, idx1)
    lm2 = get_landmark(landmarks, idx2)
    if lm1 is None or lm2 is None:
        return None
    return np.linalg.norm(lm1 - lm2)


def angle_between_landmarks(landmarks, idx1, idx2, idx3):
    """Calculate angle formed by three landmarks (idx2 is the vertex)"""
    p1 = get_landmark(landmarks, idx1)
    p2 = get_landmark(landmarks, idx2)
    p3 = get_landmark(landmarks, idx3)
    
    if p1 is None or p2 is None or p3 is None:
        return None
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle) * 180 / np.pi
    return angle


def fetch_stretch_pose_from_web(stretch_name, website_url):
    """
    Fetch stretch pose instructions from a website using Yellowcake API.
    Returns pose description or None if API key not set.
    """
    if not YELLOWCAKE_API_KEY:
        return None
    
    try:
        prompt = f"Find detailed instructions and description for how to perform the {stretch_name} stretch. Include the correct body position, alignment, and what the person should feel."
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": YELLOWCAKE_API_KEY,
        }
        
        data = {
            "url": website_url,
            "prompt": prompt,
        }
        
        print(f"Fetching {stretch_name} instructions from {website_url}...")
        
        response = requests.post(YELLOWCAKE_API_URL, json=data, headers=headers, timeout=300)
        
        if response.status_code == 200:
            # Parse the SSE stream response
            lines = response.text.strip().split('\n')
            for line in lines:
                if line.startswith('data: '):
                    try:
                        json_data = json.loads(line[6:])
                        if json_data.get('success') and json_data.get('data'):
                            # Extract the first result
                            result = json_data['data'][0] if isinstance(json_data['data'], list) else json_data['data']
                            # Return the first value as the instruction
                            if isinstance(result, dict):
                                return next(iter(result.values()))
                            else:
                                return str(result)
                    except json.JSONDecodeError:
                        continue
        
        return None
    except Exception as e:
        print(f"Error fetching from Yellowcake: {e}")
        return None


def classify_user_stretch_with_ai(landmarks, expected_stretch_name):
    """
    Use Google Gemini AI to classify which stretch the user is performing based on pose landmarks.
    Returns True if the classification matches the expected stretch, False otherwise.
    Only works if GEMINI_API_KEY is set.
    """
    if not GEMINI_API_KEY or landmarks is None:
        return None  # Return None if API key not set (fall back to rule-based verification)
    
    try:
        # Format landmark data for Claude (simplified for API limits)
        # Extract key body positions
        nose = get_landmark(landmarks, 0)
        l_shoulder = get_landmark(landmarks, 11)
        r_shoulder = get_landmark(landmarks, 12)
        l_hip = get_landmark(landmarks, 23)
        r_hip = get_landmark(landmarks, 24)
        l_knee = get_landmark(landmarks, 25)
        r_knee = get_landmark(landmarks, 26)
        l_ankle = get_landmark(landmarks, 27)
        r_ankle = get_landmark(landmarks, 28)
        l_wrist = get_landmark(landmarks, 15)
        r_wrist = get_landmark(landmarks, 16)
        l_elbow = get_landmark(landmarks, 13)
        r_elbow = get_landmark(landmarks, 14)
        
        # Create pose description
        pose_data = {
            "head": f"({nose[0]:.2f}, {nose[1]:.2f})" if nose is not None else "unknown",
            "shoulders": f"L({l_shoulder[0]:.2f}, {l_shoulder[1]:.2f}) R({r_shoulder[0]:.2f}, {r_shoulder[1]:.2f})" if l_shoulder is not None else "unknown",
            "hips": f"L({l_hip[0]:.2f}, {l_hip[1]:.2f}) R({r_hip[0]:.2f}, {r_hip[1]:.2f})" if l_hip is not None else "unknown",
            "knees": f"L({l_knee[0]:.2f}, {l_knee[1]:.2f}) R({r_knee[0]:.2f}, {r_knee[1]:.2f})" if l_knee is not None else "unknown",
            "ankles": f"L({l_ankle[0]:.2f}, {l_ankle[1]:.2f}) R({r_ankle[0]:.2f}, {r_ankle[1]:.2f})" if l_ankle is not None else "unknown",
            "wrists": f"L({l_wrist[0]:.2f}, {l_wrist[1]:.2f}) R({r_wrist[0]:.2f}, {r_wrist[1]:.2f})" if l_wrist is not None else "unknown",
        }
        
        prompt = f"""You are a fitness pose classifier. Based on the following body landmark coordinates (normalized 0-1), classify which stretch the user is performing.

Body positions (x, y normalized coordinates where 0,0 is top-left, 1,1 is bottom-right):
{json.dumps(pose_data, indent=2)}

Available stretches:
1. Cat-Cow: Alternates between arched back (COW) with hips dropped and head lifted, and rounded back (CAT) with shoulders hunched
2. Child's Pose: Forehead to ground, hips back to heels, arms extended or relaxed
3. Spinal Twist: Torso rotated, sitting with legs extended
4. Chest Opener: Standing upright, arms pulled back behind shoulders, chest open
5. Shoulder Rolls: Standing with shoulders elevated toward ears, rolling smoothly
6. Neck Side Stretch: Head tilted to one shoulder
7. Hamstring Stretch: One leg extended, bending forward at hips
8. Dynamic Lunges: One leg forward bent, one leg back bent, in lunge position
9. Shoulder Shrugs: Standing with shoulders raised up toward ears

Based on these coordinates, which stretch is the user most likely performing RIGHT NOW?
Respond with ONLY the exact stretch name from the list above, or "None" if unclear. Do not include explanation."""
        
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                headers={
                    "Content-Type": "application/json",
                },
                params={
                    "key": GEMINI_API_KEY,
                },
                json={
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "maxOutputTokens": 50,
                        "temperature": 0.1,
                    }
                },
                timeout=10
            )
            
            if DEBUG_AI:
                print(f"[DEBUG] Gemini API called for {expected_stretch_name}. Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get('candidates') and len(result['candidates']) > 0:
                        content = result['candidates'][0].get('content', {})
                        parts = content.get('parts', [])
                        if parts and len(parts) > 0:
                            ai_classification = parts[0].get('text', '').strip()
                            if ai_classification:
                                # Check if AI's classification matches expected stretch
                                is_match = expected_stretch_name.lower() in ai_classification.lower()
                                if DEBUG_AI:
                                    print(f"[DEBUG] AI Classification: {ai_classification} -> Match: {is_match}")
                                return is_match
                    return None
                except (KeyError, IndexError, TypeError) as parse_error:
                    return None  # Fall back to rule-based on parse error
            elif response.status_code == 429:
                # Rate limited - silently fail and use rule-based
                if DEBUG_AI:
                    print(f"[DEBUG] Rate limited (429) - switching to rule-based verification")
                return None
            elif response.status_code == 400:
                # Bad request - likely malformed prompt
                return None
            else:
                # Other error (including 404)
                if DEBUG_AI:
                    print(f"[DEBUG] API error {response.status_code} - using rule-based verification")
                return None
            
            return None  # Return None if API call fails (fall back to rule-based)
        except (requests.RequestException, requests.Timeout, requests.ConnectionError) as e:
            # Network error, timeout, SSL error, etc. - silently fall back to rule-based
            if DEBUG_AI:
                print(f"[DEBUG] Network error calling Gemini API: {type(e).__name__} - using rule-based verification")
            return None
    except requests.exceptions.Timeout:
        # API call timed out - silently fail
        return None
    except requests.exceptions.ConnectionError:
        # Network error - silently fail
        return None
    except Exception as e:
        # Silent fail - just use rule-based verification
        return None  # Fall back to rule-based verification


def verify_cat_cow(landmarks):
    """Verify Cat-Cow pose - alternates between arched and rounded back"""
    # Looking for spine curvature: check if head is up/down and hips are positioned correctly
    if landmarks is None:
        return False
    
    # Get key points: nose, left shoulder, right shoulder, left hip, right hip
    nose = get_landmark(landmarks, 0)
    l_shoulder = get_landmark(landmarks, 11)
    r_shoulder = get_landmark(landmarks, 12)
    l_hip = get_landmark(landmarks, 23)
    r_hip = get_landmark(landmarks, 24)
    
    if any(x is None for x in [nose, l_shoulder, r_shoulder, l_hip, r_hip]):
        return False
    
    # Check if in a horizontal position (hands and knees or similar)
    shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
    hip_y = (l_hip[1] + r_hip[1]) / 2
    
    # In cat-cow, shoulders and hips should be close in vertical position
    y_diff = abs(shoulder_y - hip_y)
    
    # Tolerance: shoulders and hips roughly aligned horizontally
    return y_diff < 0.2


def verify_childs_pose(landmarks):
    """Verify Child's Pose - forehead down, hips back, arms extended"""
    if landmarks is None:
        return False
    
    nose = get_landmark(landmarks, 0)
    l_hip = get_landmark(landmarks, 23)
    r_hip = get_landmark(landmarks, 24)
    l_wrist = get_landmark(landmarks, 15)
    r_wrist = get_landmark(landmarks, 16)
    
    if any(x is None for x in [nose, l_hip, r_hip, l_wrist, r_wrist]):
        return False
    
    hip_y = (l_hip[1] + r_hip[1]) / 2
    
    # In child's pose, nose should be lower than hips, and wrists should be forward
    nose_below_hips = nose[1] > hip_y
    wrist_forward = (l_wrist[0] + r_wrist[0]) / 2 > (l_hip[0] + r_hip[0]) / 2
    
    return nose_below_hips and wrist_forward


def verify_chest_opener(landmarks):
    """Verify Chest Opener - chest open, shoulders back"""
    if landmarks is None:
        return False
    
    l_shoulder = get_landmark(landmarks, 11)
    r_shoulder = get_landmark(landmarks, 12)
    l_hip = get_landmark(landmarks, 23)
    r_hip = get_landmark(landmarks, 24)
    
    if any(x is None for x in [l_shoulder, r_shoulder, l_hip, r_hip]):
        return False
    
    # Get arms to verify they're pulled back
    l_shoulder = get_landmark(landmarks, 11)
    r_shoulder = get_landmark(landmarks, 12)
    l_elbow = get_landmark(landmarks, 13)
    r_elbow = get_landmark(landmarks, 14)
    l_wrist = get_landmark(landmarks, 15)
    r_wrist = get_landmark(landmarks, 16)
    
    if any(x is None for x in [l_elbow, r_elbow, l_wrist, r_wrist]):
        return False
    
    # Check if shoulders are roughly level
    shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
    
    # Check if standing/upright (shoulders above hips)
    shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
    hip_y = (l_hip[1] + r_hip[1]) / 2
    
    # Arms must be pulled back significantly (elbows behind shoulders)
    shoulder_x = (l_shoulder[0] + r_shoulder[0]) / 2
    wrist_x = (l_wrist[0] + r_wrist[0]) / 2
    elbow_x = (l_elbow[0] + r_elbow[0]) / 2
    
    # Arms should be back and extended
    arms_back = elbow_x < shoulder_x - 0.1 and wrist_x < shoulder_x - 0.15
    
    return shoulder_diff < 0.15 and shoulder_y < hip_y and arms_back


def verify_hamstring_stretch(landmarks):
    """Verify Hamstring Stretch - one leg extended, bending forward"""
    if landmarks is None:
        return False
    
    l_hip = get_landmark(landmarks, 23)
    r_hip = get_landmark(landmarks, 24)
    l_knee = get_landmark(landmarks, 25)
    r_knee = get_landmark(landmarks, 26)
    nose = get_landmark(landmarks, 0)
    
    if any(x is None for x in [l_hip, r_hip, l_knee, r_knee, nose]):
        return False
    
    # One knee should be more extended than the other
    l_knee_dist = distance_between_landmarks(landmarks, 23, 25)
    r_knee_dist = distance_between_landmarks(landmarks, 24, 26)
    
    if l_knee_dist is None or r_knee_dist is None:
        return False
    
    # One leg significantly more extended
    knee_diff = abs(l_knee_dist - r_knee_dist)
    
    # Upper body folded forward
    nose_y = nose[1]
    hip_y = (l_hip[1] + r_hip[1]) / 2
    
    return knee_diff > 0.1 and nose_y > hip_y  # Folded forward


def verify_shoulder_rolls(landmarks):
    """Verify Shoulder Rolls - shoulders elevated and back"""
    if landmarks is None:
        return False
    
    l_shoulder = get_landmark(landmarks, 11)
    r_shoulder = get_landmark(landmarks, 12)
    l_ear = get_landmark(landmarks, 7)
    r_ear = get_landmark(landmarks, 8)
    
    if any(x is None for x in [l_shoulder, r_shoulder, l_ear, r_ear]):
        return False
    
    # Shoulders should be elevated (close to ears)
    l_dist = abs(l_shoulder[1] - l_ear[1])
    r_dist = abs(r_shoulder[1] - r_ear[1])
    
    # In shoulder rolls, shoulders move up close to ears
    return (l_dist < 0.2 or r_dist < 0.2)


def verify_neck_side_stretch(landmarks):
    """Verify Neck Side Stretch - head tilted to side"""
    if landmarks is None:
        return False
    
    nose = get_landmark(landmarks, 0)
    l_ear = get_landmark(landmarks, 7)
    r_ear = get_landmark(landmarks, 8)
    
    if any(x is None for x in [nose, l_ear, r_ear]):
        return False
    
    # Check if head is tilted (nose not centered between ears)
    left_ear_x = l_ear[0]
    right_ear_x = r_ear[0]
    nose_x = nose[0]
    
    center_x = (left_ear_x + right_ear_x) / 2
    
    # Head should be tilted to one side
    tilt = abs(nose_x - center_x)
    
    return tilt > 0.05


def verify_spinal_twist(landmarks):
    """Verify Spinal Twist - torso rotated"""
    if landmarks is None:
        return False
    
    l_shoulder = get_landmark(landmarks, 11)
    r_shoulder = get_landmark(landmarks, 12)
    l_hip = get_landmark(landmarks, 23)
    r_hip = get_landmark(landmarks, 24)
    
    if any(x is None for x in [l_shoulder, r_shoulder, l_hip, r_hip]):
        return False
    
    # In a twist, shoulder and hip alignment should be different (rotated)
    shoulder_center_x = (l_shoulder[0] + r_shoulder[0]) / 2
    hip_center_x = (l_hip[0] + r_hip[0]) / 2
    
    # Centers should be offset due to twist
    rotation = abs(shoulder_center_x - hip_center_x)
    
    return rotation > 0.08


def verify_dynamic_lunges(landmarks):
    """Verify Dynamic Lunges - one leg forward, one back, bent knees"""
    if landmarks is None:
        return False
    
    l_hip = get_landmark(landmarks, 23)
    r_hip = get_landmark(landmarks, 24)
    l_knee = get_landmark(landmarks, 25)
    r_knee = get_landmark(landmarks, 26)
    l_ankle = get_landmark(landmarks, 27)
    r_ankle = get_landmark(landmarks, 28)
    
    if any(x is None for x in [l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]):
        return False
    
    # Both knees should be bent
    l_knee_bend = l_knee[1] > l_hip[1]
    r_knee_bend = r_knee[1] > r_hip[1]
    
    # Feet should be apart
    feet_dist = abs(l_ankle[0] - r_ankle[0])
    
    return (l_knee_bend or r_knee_bend) and feet_dist > 0.15


# Dictionary mapping stretch names to verification functions
stretch_verifiers = {
    "Cat-Cow": verify_cat_cow,
    "Child's Pose": verify_childs_pose,
    "Spinal Twist": verify_spinal_twist,
    "Chest Opener": verify_chest_opener,
    "Shoulder Rolls": verify_shoulder_rolls,
    "Neck Side Stretch": verify_neck_side_stretch,
    "Hamstring Stretch": verify_hamstring_stretch,
    "Dynamic Lunges": verify_dynamic_lunges,
    "Shoulder Shrugs": verify_shoulder_rolls,  # Similar to shoulder rolls
}


def generate_stretch_session(goal, session_time_seconds):
    candidate_names = goal_to_stretches.get(goal, [])
    candidate_stretches = [s for s in stretches if s["name"] in candidate_names]

    random.shuffle(candidate_stretches)

    session = []
    total_time = 0

    while total_time < session_time_seconds:
        for stretch in candidate_stretches:
            # Calculate time for this stretch (reps-based use approximate time, duration-based use exact)
            stretch_time = stretch.get("duration", stretch.get("reps", 30) * 2)  # Estimate 2 sec per rep
            
            if total_time + stretch_time <= session_time_seconds:
                session.append(stretch)
                total_time += stretch_time
            if total_time >= session_time_seconds:
                break
        else:
            break

    formatted_session = []
    for s in session:
        if "duration" in s:
            formatted_session.append({"Stretch": s["name"], "Duration_sec": s["duration"]})
        else:
            # For rep-based, estimate duration (2 sec per rep)
            formatted_session.append({"Stretch": s["name"], "Duration_sec": s.get("reps", 30) * 2})

    return formatted_session


def run_interactive_stretch_session(goal, session_time_seconds, website_url=None):
    """Run an interactive stretch session with pose verification
    
    Args:
        goal: The fitness goal (e.g., "Back Pain")
        session_time_seconds: Total time for session
        website_url: Optional website URL to fetch pose instructions from using Yellowcake
    """
    
    # Generate session first to show instructions
    session = generate_stretch_session(goal, session_time_seconds)
    
    if not session:
        print(f"No stretches found for goal: {goal}")
        return
    
    # Display instructions for all stretches in this session
    print(f"\n{'='*60}")
    print(f"STRETCH INSTRUCTIONS FOR {goal.upper()}")
    print(f"{'='*60}\n")
    
    for idx, stretch in enumerate(session, 1):
        stretch_info = next((s for s in stretches if s["name"] == stretch["Stretch"]), None)
        if stretch_info:
            if "reps" in stretch_info:
                print(f"{idx}. {stretch_info['name']} ({stretch_info['reps']} reps)")
            else:
                print(f"{idx}. {stretch_info['name']} ({stretch['Duration_sec']}s)")
            
            # Try to fetch from website if URL provided
            instructions = None
            if website_url:
                instructions = fetch_stretch_pose_from_web(stretch_info["name"], website_url)
            
            # Use fetched instructions or fallback to hardcoded ones
            if instructions:
                print(f"   {instructions}\n")
            elif "instructions" in stretch_info:
                print(f"   {stretch_info['instructions']}\n")
    
    input("Press ENTER when ready to start the workout...")
    
    print("Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera opened successfully")
    print("Initializing MediaPipe Pose...")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera opened successfully")
    print("Initializing MediaPipe Pose...")
    
    print(f"\n{'='*50}")
    print(f"Starting {goal} workout session!")
    print(f"{'='*50}\n")

    stretch_idx = 0
    current_stretch = session[stretch_idx]
    
    # Get full stretch info including reps/duration
    current_stretch_info = next((s for s in stretches if s["name"] == current_stretch["Stretch"]), None)
    is_rep_based = "reps" in current_stretch_info
    
    if is_rep_based:
        reps_remaining = current_stretch_info["reps"]
        reps_completed = 0
    else:
        remaining_time = current_stretch["Duration_sec"]
    
    last_time = time.time()
    in_correct_pose = False
    pose_confirmed_time = 0
    last_pose_state = False
    
    frame_count = 0
    ai_check_interval = 30
    cached_ai_result = None
    last_api_call_time = 0
    min_api_call_delay = 2.0

    while stretch_idx < len(session):
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            landmarks = extract_pose_data(results)
            
            # Only call AI every N frames to reduce lag (instead of every frame)
            # Also enforce minimum delay between API calls to avoid rate limiting
            api_call_time = time.time()
            if frame_count % ai_check_interval == 0 and (api_call_time - last_api_call_time) >= min_api_call_delay:
                ai_result = classify_user_stretch_with_ai(landmarks, current_stretch["Stretch"])
                if ai_result is not None:
                    cached_ai_result = ai_result
                last_api_call_time = api_call_time
            
            # Use cached AI result or fall back to rule-based verification
            if cached_ai_result is not None:
                in_correct_pose = cached_ai_result
            else:
                # Fall back to rule-based verification
                verifier = stretch_verifiers.get(current_stretch["Stretch"])
                if verifier and landmarks is not None:
                    in_correct_pose = verifier(landmarks)
        else:
            in_correct_pose = False

        # Handle time-based and rep-based exercises differently
        current_time = time.time()
        
        if is_rep_based:
            # For rep-based: count a rep when transitioning from not-in-pose to in-pose
            if in_correct_pose and not last_pose_state:
                reps_completed += 1
                reps_remaining = max(0, current_stretch_info["reps"] - reps_completed)
            last_pose_state = in_correct_pose
        else:
            # For time-based: only count down timer if in correct pose
            if in_correct_pose:
                time_elapsed = current_time - last_time
                remaining_time -= time_elapsed
                pose_confirmed_time += time_elapsed
        
        last_time = current_time

        # Display information on frame
        frame_height, frame_width = frame.shape[:2]
        
        # Stretch name
        cv2.putText(
            frame,
            f"Stretch: {current_stretch['Stretch']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        
        # Timer or Reps counter
        timer_color = (0, 255, 0) if in_correct_pose else (0, 0, 255)
        
        if is_rep_based:
            counter_text = f"Reps: {reps_completed}/{current_stretch_info['reps']}"
        else:
            counter_text = f"Time: {max(0, int(remaining_time))}s"
        
        cv2.putText(
            frame,
            counter_text,
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            timer_color,
            2,
        )
        
        # Pose status with verification method indicator
        if GEMINI_API_KEY:
            status_text = "✓ CORRECT POSE (AI)" if in_correct_pose else "✗ Adjust position (AI)"
        else:
            status_text = "✓ CORRECT POSE" if in_correct_pose else "✗ Adjust position"
        status_color = (0, 255, 0) if in_correct_pose else (0, 0, 255)
        cv2.putText(
            frame,
            status_text,
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2,
        )

        # Progress
        progress_text = f"{stretch_idx + 1}/{len(session)}"
        cv2.putText(
            frame,
            progress_text,
            (frame_width - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Workout Session", frame)

        # Check if stretch is complete
        stretch_complete = False
        if is_rep_based:
            stretch_complete = reps_completed >= current_stretch_info["reps"]
        else:
            stretch_complete = remaining_time <= 0
        
        if stretch_complete:
            print(f"✓ Completed: {current_stretch['Stretch']}")
            stretch_idx += 1
            
            if stretch_idx < len(session):
                current_stretch = session[stretch_idx]
                current_stretch_info = next((s for s in stretches if s["name"] == current_stretch["Stretch"]), None)
                is_rep_based = "reps" in current_stretch_info
                
                if is_rep_based:
                    reps_remaining = current_stretch_info["reps"]
                    reps_completed = 0
                    print(f"\nNext stretch: {current_stretch['Stretch']} ({current_stretch_info['reps']} reps)")
                else:
                    remaining_time = current_stretch["Duration_sec"]
                    print(f"\nNext stretch: {current_stretch['Stretch']} ({current_stretch['Duration_sec']}s)")
                
                print("Get into position...\n")
                pose_confirmed_time = 0
                last_pose_state = False
                cached_ai_result = None  # Reset cache for new stretch
            
            last_time = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nWorkout interrupted by user")
            break
        
        frame_count += 1  # Increment frame counter for throttling

    pose.close()
    cap.release()
    cv2.destroyAllWindows()

    if stretch_idx == len(session):
        print(f"\n{'='*50}")
        print("✓ Workout complete! Great job!")
        print(f"{'='*50}\n")
    else:
        print(f"\nWorkout stopped. Completed {stretch_idx}/{len(session)} stretches.")


if __name__ == "__main__":
    goal = "Desk Break"
    session_length_min = 5
    session_length_sec = session_length_min * 60

    if GEMINI_API_KEY:
        print("✓ AI Pose Classification ENABLED (using Google Gemini)")
    else:
        print("ℹ AI Pose Classification DISABLED (using rule-based verification)")
        print("  To enable: export GEMINI_API_KEY=\"your-api-key-here\"\n")

    website_url = None
    
    run_interactive_stretch_session(goal, session_length_sec, website_url=website_url)
