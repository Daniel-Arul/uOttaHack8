import numpy as np
import mediapipe as mp

def extract_pose_data(results):
    if not results.pose_landmarks:
        return None

    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])

    return np.array(landmarks)

def get_landmark_names():
    """Get all 33 landmark names from MediaPipe"""
    mp_pose = mp.solutions.pose
    landmark_names = []
    for landmark in mp_pose.PoseLandmark:
        landmark_names.append(landmark.name)
    return landmark_names

def analyze_posture(base_data, current_data, threshold=0.035):
    """Analyze posture deviations from baseline data"""
    if base_data is None or current_data is None:
        return None
    
    base_array = np.array(base_data)
    current_array = np.array(current_data)
    
    # MediaPipe landmark indices
    LEFT_SHOULDER = 11 * 3
    RIGHT_SHOULDER = 12 * 3
    LEFT_HIP = 23 * 3
    RIGHT_HIP = 24 * 3
    NOSE = 0 * 3
    LEFT_EAR = 3 * 3
    RIGHT_EAR = 4 * 3
    
    issues = []
    
    # Check for uneven shoulders
    left_shoulder_y = current_array[LEFT_SHOULDER + 1]
    right_shoulder_y = current_array[RIGHT_SHOULDER + 1]
    shoulder_diff = abs(left_shoulder_y - right_shoulder_y)
    
    if shoulder_diff > threshold:
        issues.append({
            "type": "Uneven Shoulders",
            "severity": shoulder_diff,
            "description": "Muscle imbalance detected"
        })
    
    # Check for torso off-center (hip and shoulder alignment)
    left_hip_x = current_array[LEFT_HIP]
    right_hip_x = current_array[RIGHT_HIP]
    left_shoulder_x = current_array[LEFT_SHOULDER]
    right_shoulder_x = current_array[RIGHT_SHOULDER]
    
    hip_center_x = (left_hip_x + right_hip_x) / 2
    shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
    torso_offset = abs(hip_center_x - shoulder_center_x)
    
    if torso_offset > threshold:
        issues.append({
            "type": "Torso Off-Center",
            "severity": torso_offset,
            "description": "Core or pelvic asymmetry detected"
        })
    
    # Check for head tilt
    nose_x = current_array[NOSE]
    left_ear_x = current_array[LEFT_EAR]
    right_ear_x = current_array[RIGHT_EAR]
    
    nose_center = (left_ear_x + right_ear_x) / 2
    head_tilt = abs(nose_x - nose_center)
    
    if head_tilt > threshold:
        issues.append({
            "type": "Head Tilt",
            "severity": head_tilt,
            "description": "Neck compensation detected"
        })
    
    # Check for slouching using vertical distance between face and shoulders
    # Get current landmarks
    nose_current = current_array[NOSE:NOSE + 3]
    left_shoulder_current = current_array[LEFT_SHOULDER:LEFT_SHOULDER + 3]
    right_shoulder_current = current_array[RIGHT_SHOULDER:RIGHT_SHOULDER + 3]
    
    # Get baseline landmarks
    nose_base = base_array[NOSE:NOSE + 3]
    left_shoulder_base = base_array[LEFT_SHOULDER:LEFT_SHOULDER + 3]
    right_shoulder_base = base_array[RIGHT_SHOULDER:RIGHT_SHOULDER + 3]
    
    # Calculate average shoulder Y position
    shoulder_y_current = (left_shoulder_current[1] + right_shoulder_current[1]) / 2
    shoulder_y_base = (left_shoulder_base[1] + right_shoulder_base[1]) / 2
    
    # Get nose Y position
    nose_y_current = nose_current[1]
    nose_y_base = nose_base[1]
    
    # Calculate vertical distances (larger Y = lower on screen)
    # Distance = nose Y - shoulder Y (positive when face is above shoulders)
    distance_current = nose_y_current - shoulder_y_current
    distance_base = nose_y_base - shoulder_y_base
    
    # Calculate change in distance
    if distance_base != 0:
        distance_change = (distance_current - distance_base) / abs(distance_base)
    else:
        distance_change = 0
    
    slouch_threshold = 0.05  # 5% threshold
    
    if distance_change < -slouch_threshold or distance_change > slouch_threshold:
        # Big enough change = head moved too close or too far from shoulders = slouching
        issues.append({
            "type": "Slouching",
            "severity": abs(distance_change),
            "description": f"Head moving toward or away from shoulders (change: {distance_change:.3f})"
        })
    
    return issues