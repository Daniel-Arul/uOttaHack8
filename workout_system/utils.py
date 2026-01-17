"""Utility functions for pose processing"""

import numpy as np


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
