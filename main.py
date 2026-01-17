import cv2 
import numpy as np
import mediapipe as mp
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

def extract_pose_data(self, results):
        if not results.pose_landmarks:
            return None

        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

        return np.array(landmarks)

def classify():

    cap = cv2.VideoCapture(0)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    while True:
        
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

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()