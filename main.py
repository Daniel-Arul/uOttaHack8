import cv2 
import numpy as np
import mediapipe as mp
import os
import csv
from datetime import datetime
from helpers import extract_pose_data, get_landmark_names, analyze_posture
from calibrate import calibrate

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

csv_data = []

def classify(data):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 6)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"pose_data_{timestamp}.csv"
    
    frame_count = 0

    while True:
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
            
            # Analyze posture and detect issues
            issues = analyze_posture(data, landmarks.tolist())
            
            if issues:
                y_offset = 30
                for issue in issues:
                    cv2.putText(frame, f"{issue['type']}: {issue['severity']:.3f}", 
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y_offset += 25

        cv2.imshow("frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    pose.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    base_data = calibrate()
    classify(base_data)