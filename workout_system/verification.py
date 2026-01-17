"""Pose verification functions for detecting correct stretch positions"""

from .utils import get_landmark, distance_between_landmarks


def verify_cat_cow(landmarks):
    """Verify Cat-Cow pose - alternates between arched and rounded back"""
    if landmarks is None:
        return False
    
    nose = get_landmark(landmarks, 0)
    l_shoulder = get_landmark(landmarks, 11)
    r_shoulder = get_landmark(landmarks, 12)
    l_hip = get_landmark(landmarks, 23)
    r_hip = get_landmark(landmarks, 24)
    
    if any(x is None for x in [nose, l_shoulder, r_shoulder, l_hip, r_hip]):
        return False
    
    shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
    hip_y = (l_hip[1] + r_hip[1]) / 2
    y_diff = abs(shoulder_y - hip_y)
    
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
    l_elbow = get_landmark(landmarks, 13)
    r_elbow = get_landmark(landmarks, 14)
    l_wrist = get_landmark(landmarks, 15)
    r_wrist = get_landmark(landmarks, 16)
    
    if any(x is None for x in [l_shoulder, r_shoulder, l_hip, r_hip, l_elbow, r_elbow, l_wrist, r_wrist]):
        return False
    
    shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
    shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
    hip_y = (l_hip[1] + r_hip[1]) / 2
    
    shoulder_x = (l_shoulder[0] + r_shoulder[0]) / 2
    wrist_x = (l_wrist[0] + r_wrist[0]) / 2
    elbow_x = (l_elbow[0] + r_elbow[0]) / 2
    
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
    
    l_knee_dist = distance_between_landmarks(landmarks, 23, 25)
    r_knee_dist = distance_between_landmarks(landmarks, 24, 26)
    
    if l_knee_dist is None or r_knee_dist is None:
        return False
    
    knee_diff = abs(l_knee_dist - r_knee_dist)
    nose_y = nose[1]
    hip_y = (l_hip[1] + r_hip[1]) / 2
    
    return knee_diff > 0.1 and nose_y > hip_y


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
    
    l_dist = abs(l_shoulder[1] - l_ear[1])
    r_dist = abs(r_shoulder[1] - r_ear[1])
    
    return (l_dist < 0.25 or r_dist < 0.25)


def verify_neck_side_stretch(landmarks):
    """Verify Neck Side Stretch - head tilted to side"""
    if landmarks is None:
        return False
    
    nose = get_landmark(landmarks, 0)
    l_ear = get_landmark(landmarks, 7)
    r_ear = get_landmark(landmarks, 8)
    
    if any(x is None for x in [nose, l_ear, r_ear]):
        return False
    
    left_ear_x = l_ear[0]
    right_ear_x = r_ear[0]
    nose_x = nose[0]
    center_x = (left_ear_x + right_ear_x) / 2
    
    tilt = abs(nose_x - center_x)
    
    return tilt > 0.03


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
    
    shoulder_center_x = (l_shoulder[0] + r_shoulder[0]) / 2
    hip_center_x = (l_hip[0] + r_hip[0]) / 2
    rotation = abs(shoulder_center_x - hip_center_x)
    
    return rotation > 0.05


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
    
    l_knee_bend = l_knee[1] > l_hip[1]
    r_knee_bend = r_knee[1] > r_hip[1]
    feet_dist = abs(l_ankle[0] - r_ankle[0])
    
    return (l_knee_bend or r_knee_bend) and feet_dist > 0.15


# Dictionary mapping stretch names to verification functions
stretch_verifiers = {
    "Arm Crossover Stretch": verify_chest_opener,
    "Wrist Rotations": verify_shoulder_rolls,
    "Reach Behind the Back": verify_chest_opener,
    "Neck Tilts": verify_neck_side_stretch,
    "Side Bends": verify_spinal_twist,
    "Chest Opener": verify_chest_opener,
    "Spinal Twist": verify_spinal_twist,
    "Shoulder Rolls": verify_shoulder_rolls,
    "Neck Rolls": verify_neck_side_stretch,
}
