"""AI-powered pose classification and web-based instruction fetching"""

import json
import requests
from .config import YELLOWCAKE_API_KEY, YELLOWCAKE_API_URL, GEMINI_API_KEY, DEBUG_AI
from .utils import get_landmark


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
