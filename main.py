import cv2 
import numpy as np
import mediapipe as mp
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

csv_data = []

def extract_pose_data(results):
    if not results.pose_landmarks:
        return None

    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])

    

    return np.array(landmarks)

def classify():
    print("Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera opened successfully")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    print("Initializing MediaPipe Pose...")
    pose = mp_pose.Pose()
    print("MediaPipe Pose initialized")

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

        cv2.imshow("frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    pose.close()  # Properly close MediaPipe
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    classify()