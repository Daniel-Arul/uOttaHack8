import cv2
import mediapipe as mp
from helpers import extract_pose_data

# Function that runs for calibration, returns an array of length 99 corresponding to the x,y,z positions of different body components
def calibrate(cap):
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose()
    
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        landmarks = extract_pose_data(results)
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return landmarks.tolist()
