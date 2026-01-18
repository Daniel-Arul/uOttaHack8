"""Session generation and interactive workout session management"""

import random
import cv2
import time
import mediapipe as mp
import winsound
from .stretches_db import stretches, goal_to_stretches
from .utils import extract_pose_data
from .ai import fetch_stretch_pose_from_web, classify_user_stretch_with_ai
from .verification import stretch_verifiers
from .config import GEMINI_API_KEY, DEBUG_AI, AI_CHECK_INTERVAL, MIN_API_CALL_DELAY


def run_interactive_stretch_session_qt(goal, session_time_seconds, frame_callback, should_stop_callback, website_url=None):
    """Run an interactive stretch session with pose verification for Qt integration
    
    Args:
        goal: The fitness goal (e.g., "Back Pain")
        session_time_seconds: Total time for session
        frame_callback: Callback function to send frames to (receives BGR frame)
        should_stop_callback: Callback function to check if workout should stop (returns bool)
        website_url: Optional website URL to fetch pose instructions from using Yellowcake
    
    Returns:
        True if workout completed successfully, False if interrupted
    """

    # Generate session first to show instructions
    session = generate_stretch_session(goal, session_time_seconds)

    if not session:
        return False

    # Open camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        return False

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    stretch_idx = 0
    current_stretch = session[stretch_idx]
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
    cached_ai_result = None
    last_api_call_time = 0
    rep_cooldown = 0
    REP_COOLDOWN_DURATION = 0.5  # seconds between reps to avoid double-counting

    while stretch_idx < len(session) and not should_stop_callback():
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            landmarks = extract_pose_data(results)

            api_call_time = time.time()
            if frame_count % AI_CHECK_INTERVAL == 0 and (api_call_time - last_api_call_time) >= MIN_API_CALL_DELAY:
                ai_result = classify_user_stretch_with_ai(landmarks, current_stretch["Stretch"])
                if ai_result is not None:
                    cached_ai_result = ai_result
                last_api_call_time = api_call_time

            if cached_ai_result is not None:
                in_correct_pose = cached_ai_result
            else:
                verifier = stretch_verifiers.get(current_stretch["Stretch"])
                if verifier and landmarks is not None:
                    in_correct_pose = verifier(landmarks)
        else:
            in_correct_pose = False

        current_time = time.time()

        # Update cooldown
        if rep_cooldown > 0:
            rep_cooldown -= (current_time - last_time)

        if is_rep_based:
            if in_correct_pose and not last_pose_state and rep_cooldown <= 0:
                reps_completed += 1
                rep_cooldown = REP_COOLDOWN_DURATION  # Start cooldown after counting rep
                reps_remaining = max(0, current_stretch_info["reps"] - reps_completed)
            last_pose_state = in_correct_pose
        else:
            if in_correct_pose:
                time_elapsed = current_time - last_time
                remaining_time -= time_elapsed
                pose_confirmed_time += time_elapsed

        last_time = current_time

        # Display information on frame
        frame_height, frame_width = frame.shape[:2]

        cv2.putText(
            frame,
            f"Stretch: {current_stretch['Stretch']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 150, 100),
            2,
        )

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

        # Send frame to Qt callback instead of cv2.imshow
        frame_callback(frame)

        stretch_complete = False
        if is_rep_based:
            stretch_complete = reps_completed >= current_stretch_info["reps"]
        else:
            stretch_complete = remaining_time <= 0

        if stretch_complete:
            play_completion_sound()
            stretch_idx += 1

            if stretch_idx < len(session):
                current_stretch = session[stretch_idx]
                current_stretch_info = next((s for s in stretches if s["name"] == current_stretch["Stretch"]), None)
                is_rep_based = "reps" in current_stretch_info

                if is_rep_based:
                    reps_remaining = current_stretch_info["reps"]
                    reps_completed = 0
                    rep_cooldown = 0  # Reset cooldown for new stretch
                else:
                    remaining_time = current_stretch["Duration_sec"]

                pose_confirmed_time = 0
                last_pose_state = False
                cached_ai_result = None

            last_time = time.time()

        frame_count += 1

    pose.close()
    cap.release()

    return stretch_idx == len(session)


def play_completion_sound():
    try:
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except Exception:
        pass


def generate_stretch_session(goal, session_time_seconds):
    """Generate a randomized stretch session based on goal and time limit"""
    candidate_names = goal_to_stretches.get(goal, [])
    candidate_stretches = [s for s in stretches if s["name"] in candidate_names]

    random.shuffle(candidate_stretches)

    session = []
    total_time = 0

    while total_time < session_time_seconds:
        for stretch in candidate_stretches:
            stretch_time = stretch.get("duration", stretch.get("reps", 30) * 2)
            
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
            
            instructions = None
            if website_url:
                instructions = fetch_stretch_pose_from_web(stretch_info["name"], website_url)
            
            if instructions:
                print(f"   {instructions}\n")
            elif "instructions" in stretch_info:
                print(f"   {stretch_info['instructions']}\n")
    
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
    
    print(f"\n{'='*50}")
    print(f"Starting {goal} workout session!")
    print(f"{'='*50}\n")

    stretch_idx = 0
    current_stretch = session[stretch_idx]
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
    cached_ai_result = None
    last_api_call_time = 0
    rep_cooldown = 0
    REP_COOLDOWN_DURATION = 1  # seconds between reps to avoid double-counting

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
            
            api_call_time = time.time()
            if frame_count % AI_CHECK_INTERVAL == 0 and (api_call_time - last_api_call_time) >= MIN_API_CALL_DELAY:
                ai_result = classify_user_stretch_with_ai(landmarks, current_stretch["Stretch"])
                if ai_result is not None:
                    cached_ai_result = ai_result
                last_api_call_time = api_call_time
            
            if cached_ai_result is not None:
                in_correct_pose = cached_ai_result
            else:
                verifier = stretch_verifiers.get(current_stretch["Stretch"])
                if verifier and landmarks is not None:
                    in_correct_pose = verifier(landmarks)
        else:
            in_correct_pose = False

        current_time = time.time()
        
        # Update cooldown
        if rep_cooldown > 0:
            rep_cooldown -= (current_time - last_time)
        
        if is_rep_based:
            if in_correct_pose and not last_pose_state and rep_cooldown <= 0:
                reps_completed += 1
                rep_cooldown = REP_COOLDOWN_DURATION  # Start cooldown after counting rep
                reps_remaining = max(0, current_stretch_info["reps"] - reps_completed)
            last_pose_state = in_correct_pose
        else:
            if in_correct_pose:
                time_elapsed = current_time - last_time
                remaining_time -= time_elapsed
                pose_confirmed_time += time_elapsed
        
        last_time = current_time

        # Display information on frame
        frame_height, frame_width = frame.shape[:2]
        
        cv2.putText(
            frame,
            f"Stretch: {current_stretch['Stretch']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 150, 100),
            2,
        )
        
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

        stretch_complete = False
        if is_rep_based:
            stretch_complete = reps_completed >= current_stretch_info["reps"]
        else:
            stretch_complete = remaining_time <= 0
        
        if stretch_complete:
            play_completion_sound()
            print(f"✓ Completed: {current_stretch['Stretch']}")
            stretch_idx += 1
            
            if stretch_idx < len(session):
                current_stretch = session[stretch_idx]
                current_stretch_info = next((s for s in stretches if s["name"] == current_stretch["Stretch"]), None)
                is_rep_based = "reps" in current_stretch_info
                
                if is_rep_based:
                    reps_remaining = current_stretch_info["reps"]
                    reps_completed = 0
                    rep_cooldown = 0  # Reset cooldown for new stretch
                    print(f"\nNext stretch: {current_stretch['Stretch']} ({current_stretch_info['reps']} reps)")
                else:
                    remaining_time = current_stretch["Duration_sec"]
                    print(f"\nNext stretch: {current_stretch['Stretch']} ({current_stretch['Duration_sec']}s)")
                
                print("Get into position...\n")
                pose_confirmed_time = 0
                last_pose_state = False
                cached_ai_result = None
            
            last_time = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nWorkout interrupted by user")
            break
        
        frame_count += 1

    pose.close()
    cap.release()
    cv2.destroyAllWindows()

    if stretch_idx == len(session):
        print(f"\n{'='*50}")
        print("✓ Workout complete! Great job!")
        print(f"{'='*50}\n")
    else:
        print(f"\nWorkout stopped. Completed {stretch_idx}/{len(session)} stretches.")
