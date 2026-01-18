import os
import threading
from time import time

import plyer

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import sys
import cv2
import numpy as np
import mediapipe as mp

from PySide6.QtCore import QThread, Signal, Qt, QObject, QCoreApplication
from PySide6.QtGui import QImage, QPixmap, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QListWidget,
    QFrame, QSpinBox
)

from helpers import extract_pose_data, analyze_posture
from workout_system.main import main as workout_main
from workout_system.session import run_interactive_stretch_session_qt

TIMER_DURATION = 30 
# OpenCV to Qt ****************************************************************
def bgr_to_qimage(frame_bgr):
    h, w, ch = frame_bgr.shape
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


# Calibration Worker (NO cv2.imshow) ******************************************
class CalibrationWorker(QThread):
    frame_ready = Signal(QImage)
    status = Signal(str)
    done = Signal(list)        # base_data list (length 99)
    error = Signal(str)

    def __init__(self, camera_backend, fps=30, seconds=5, camera_index=0):
        super().__init__()
        self.camera_backend = camera_backend
        self.fps = fps
        self.seconds = seconds
        self.camera_index = camera_index
        self._running = False

    def stop(self):
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, self.camera_backend)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            self.error.emit("Calibration: could not open camera.")
            return

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        frames_needed = max(1, int(self.seconds * self.fps))
        collected = []

        self._running = True
        self.status.emit(f"Calibrating… sit upright ({self.seconds}s)")

        frames_seen = 0
        while self._running and frames_seen < frames_needed:
            ret, frame = cap.read()
            if not ret:
                self.error.emit("Calibration: could not read frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # We only collect if landmarks exist
            if results.pose_landmarks:
                landmarks = extract_pose_data(results)
                # landmarks.tolist() should be length 99
                collected.append(landmarks.tolist())

            # show preview inside Qt
            self.frame_ready.emit(bgr_to_qimage(frame))

            frames_seen += 1
            # small throttle; rely on your fps cap too
            self.msleep(int(1000 / max(1, self.fps)))

        cap.release()
        try:
            pose.close()
        except Exception:
            pass

        if not collected:
            self.error.emit("Calibration failed: no pose detected. Try better lighting / full body in frame.")
            return

        # Average across collected frames
        base_data = np.mean(np.array(collected, dtype=float), axis=0).tolist()
        self.status.emit("Calibration complete.")
        self.done.emit(base_data)


# Posture Worker (NO cv2.imshow) *************************************************
class PostureWorker(QThread):
    frame_ready = Signal(QImage)
    issues_ready = Signal(list)
    status = Signal(str)
    error = Signal(str)

    def __init__(self, base_data, camera_backend, main_window, fps=30, camera_index=0):
        super().__init__()
        self.base_data = base_data
        self.camera_backend = camera_backend
        self.fps = fps
        self.camera_index = camera_index
        self._running = False
        self.notifier = Notification(main_window)

    def stop(self):
        self._running = False

    def run(self):
        if self.base_data is None:
            self.error.emit("Run: please calibrate first.")
            return

        cap = cv2.VideoCapture(self.camera_index, self.camera_backend)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            self.error.emit("Run: could not open camera.")
            return

        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose()

        self._running = True
        self.status.emit("Running posture monitor…")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.error.emit("Run: could not read frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            issues = []
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
                landmarks = extract_pose_data(results)
                issues = analyze_posture(self.base_data, landmarks.tolist())

                # Keep your on-frame red text overlay
                if issues:
                    y_offset = 30
                    for issue in issues:
                        cv2.putText(
                            frame,
                            f"{issue['type']}: {issue['severity']:.3f}",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )
                        y_offset += 25
                    # Start/continue the timer when issues are detected
                    self.notifier.decrement_posture_timer()
                else:
                    # Reset timer when no issues
                    self.notifier.reset_timer()

            self.frame_ready.emit(bgr_to_qimage(frame))
            self.issues_ready.emit(issues)

            self.msleep(int(1000 / max(1, self.fps)))

        cap.release()
        try:
            pose.close()
        except Exception:
            pass

        self.status.emit("Stopped.")


# Workout Worker (runs in same window) *****************************
class WorkoutWorker(QThread):
    frame_ready = Signal(QImage)
    status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, goal, session_time_seconds, camera_backend, fps=30, camera_index=0):
        super().__init__()
        self.goal = goal
        self.session_time_seconds = session_time_seconds
        self.camera_backend = camera_backend
        self.fps = fps
        self.camera_index = camera_index
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        def frame_callback(frame_bgr):
            """Callback to send frames to Qt"""
            self.frame_ready.emit(bgr_to_qimage(frame_bgr))
            # Small delay to respect FPS
            self.msleep(int(1000 / max(1, self.fps)))

        def should_stop_callback():
            """Callback to check if workout should stop"""
            return self._stop_requested

        try:
            self.status.emit(f"Starting {self.goal} workout...")
            completed = run_interactive_stretch_session_qt(
                self.goal,
                self.session_time_seconds,
                frame_callback,
                should_stop_callback
            )
            
            if completed:
                self.status.emit("Workout complete! Great job!")
            else:
                self.status.emit("Workout stopped.")
            
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"Workout error: {str(e)}")
            self.finished.emit()


# ---------- Main Window ----------
class MainWindow(QMainWindow):
    def __init__(self, camera_backend, font_family):
        super().__init__()
        self.setWindowTitle("Posture Monitor")
        self.resize(1200, 700)

        self.camera_backend = camera_backend
        self.base_data = None
        self.font_family = font_family

        self.calib_worker = None
        self.posture_worker = None
        self.workout_worker = None

        # Video preview
        self.video = QLabel("Camera preview")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(860, 560)
        self.video.setObjectName("VideoPanel")

        # Issues list
        self.issues_list = QListWidget()
        self.issues_list.setObjectName("IssuesList")

        # Status line
        self.status_label = QLabel("Click CALIBRATE to begin· Ensure head and shoulders are in the frame·")
        self.status_label.setObjectName("StatusLabel")

        # Controls
        self.calibrate_btn = QPushButton("Calibrate")
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        self.seconds_spin = QSpinBox()
        self.seconds_spin.setRange(2, 15)
        self.seconds_spin.setValue(5)
        self.seconds_spin.setSuffix("s")

        # Right panel
        right_layout = QVBoxLayout()
        title = QLabel("Posture Issues")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        right_layout.addWidget(title)
        right_layout.addWidget(self.issues_list)

        right_frame = QFrame()
        right_frame.setObjectName("RightPanel")
        right_frame.setLayout(right_layout)

        # Top layout
        top = QHBoxLayout()
        top.addWidget(self.video, stretch=3)
        top.addWidget(right_frame, stretch=1)

        # Controls layout
        controls = QHBoxLayout()
        controls.addWidget(self.calibrate_btn)

        label = QLabel("Calibration:")
        label.setStyleSheet("color: #3e5374;")
        controls.addWidget(label)

        controls.addWidget(self.seconds_spin)
        controls.addSpacing(12)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch()

        root = QVBoxLayout()
        root.addLayout(top)
        root.addWidget(self.status_label)
        root.addLayout(controls)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        # Styling
        self.setStyleSheet(f"""
            * {{ font-family: "{self.font_family}"; }}
            QMainWindow {{ background: #FEECD0; color: #506b95; font-family: "{self.font_family}"; font-size: 14px;}}
            #VideoPanel {{ background: #CCD4B1; border-radius: 18px; }}
            #RightPanel {{ background: #DCA278; border-radius: 18px; padding: 14px; margin-left: 16px; }}
            #IssuesList {{ background: #0b1224; border: 1px solid #22304a; border-radius: 12px; padding: 8px; }}
            #StatusLabel {{ padding: 12px 8px; color: #3e5374; }}
            QPushButton {{ background: #3e5374; border: 1px solid #334155; padding: 10px 14px; border-radius: 12px; font-family: "{self.font_family}";}}
            QPushButton:hover {{ background: #506b95; font-family: "{self.font_family}"; }}
            QPushButton:disabled {{ opacity: 0.5; font-family: "{self.font_family}"; }}
            QSpinBox {{ background: #3e5374; border: 1px solid #22304a; border-radius: 6px; padding: 10px; }}
        """)

        # Wiring
        self.calibrate_btn.clicked.connect(self.start_calibration)
        self.start_btn.clicked.connect(self.start_posture)
        self.stop_btn.clicked.connect(self.stop_posture)

    # ----- UI slots -----
    def on_frame(self, qimg):
        pix = QPixmap.fromImage(qimg)
        self.video.setPixmap(pix.scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def on_issues(self, issues):
        self.issues_list.clear()
        if not issues:
            self.issues_list.addItem("No issues detected·")
            return
        for issue in issues:
            item_str = f"{issue['severity']:.3f}"
            item_str = item_str.split(".")
            item_str = item_str[0] + "·" + item_str[1]
            self.issues_list.addItem(f"{issue['type']}  |  severity: {item_str}")

    def set_status(self, msg):
        self.status_label.setText(msg)

    def on_error(self, msg):
        self.set_status(msg)

    # ----- Calibration -----
    def start_calibration(self):
        # Stop posture run if active
        self.stop_posture()

        # Avoid double-start
        if self.calib_worker is not None:
            return

        self.calibrate_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.issues_list.clear()

        seconds = int(self.seconds_spin.value())

        self.calib_worker = CalibrationWorker(
            camera_backend=self.camera_backend,
            fps=30,
            seconds=seconds,
            camera_index=0
        )
        self.calib_worker.frame_ready.connect(self.on_frame)
        self.calib_worker.status.connect(self.set_status)
        self.calib_worker.error.connect(self.on_calib_error)
        self.calib_worker.done.connect(self.on_calib_done)
        self.calib_worker.start()

    def on_calib_done(self, base_data):
        self.base_data = base_data
        self.calib_worker = None

        self.set_status("Calibration complete· Press Start·")
        self.calibrate_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_calib_error(self, msg):
        self.calib_worker = None
        self.set_status(msg)
        self.calibrate_btn.setEnabled(True)
        self.start_btn.setEnabled(self.base_data is not None)
        self.stop_btn.setEnabled(False)

    # ----- Posture run -----
    def start_posture(self):
        if self.base_data is None:
            self.set_status("Please calibrate first.")
            return
        if self.posture_worker is not None:
            return

        self.calibrate_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.posture_worker = PostureWorker(
            base_data=self.base_data,
            camera_backend=self.camera_backend,
            main_window=self,
            fps=30,
            camera_index=0
        )
        self.posture_worker.frame_ready.connect(self.on_frame)
        self.posture_worker.issues_ready.connect(self.on_issues)
        self.posture_worker.status.connect(self.set_status)
        self.posture_worker.error.connect(self.on_run_error)
        self.posture_worker.finished.connect(self.on_run_finished)
        self.posture_worker.notifier = Notification(self)
        self.posture_worker.start()

    def stop_posture(self):
        if self.posture_worker:
            self.posture_worker.stop()
            self.posture_worker.wait()
            self.posture_worker = None

        self.calibrate_btn.setEnabled(True)
        self.start_btn.setEnabled(self.base_data is not None)
        self.stop_btn.setEnabled(False)

    def start_workout(self, goal):
        """Start a workout in the same window"""
        print(f"[WORKOUT] start_workout called with goal: {goal}")
        if self.workout_worker is not None:
            print("[WORKOUT] Workout already running, ignoring request")
            return

        # Stop posture monitoring
        self.stop_posture()
        
        self.calibrate_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.workout_worker = WorkoutWorker(
            goal=goal,
            session_time_seconds=60,  # 1 minute workout
            camera_backend=self.camera_backend,
            fps=30,
            camera_index=0
        )
        self.workout_worker.frame_ready.connect(self.on_frame)
        self.workout_worker.status.connect(self.set_status)
        self.workout_worker.error.connect(self.on_workout_error)
        self.workout_worker.finished.connect(self.on_workout_finished)
        print("[WORKOUT] Starting workout worker...")
        self.workout_worker.start()

    def trigger_workout_from_notification(self):
        """Slot to trigger workout from notification thread"""
        print("[WORKOUT] Triggering workout from notification...")
        self.start_workout("Back Pain")

    def stop_workout(self):
        if self.workout_worker:
            self.workout_worker.request_stop()
            self.workout_worker.wait()
            self.workout_worker = None

    def on_workout_error(self, msg):
        self.set_status(msg)
        self.stop_workout()

    def on_workout_finished(self):
        self.workout_worker = None
        # Reset the notifier's working out flag so timer can work again
        if self.posture_worker and self.posture_worker.notifier:
            self.posture_worker.notifier.workingOut = False
        # Automatically resume posture monitoring
        self.start_posture()

    def on_run_error(self, msg):
        self.set_status(msg)
        self.stop_posture()

    def on_run_finished(self):
        self.posture_worker = None
        self.calibrate_btn.setEnabled(True)
        self.start_btn.setEnabled(self.base_data is not None)
        self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        # stop threads on exit
        if self.calib_worker:
            self.calib_worker.stop()
            self.calib_worker.wait()
            self.calib_worker = None
        if self.posture_worker:
            self.posture_worker.stop()
            self.posture_worker.wait()
            self.posture_worker = None
        if self.workout_worker:
            self.workout_worker.request_stop()
            self.workout_worker.wait()
            self.workout_worker = None
        event.accept()

class Notification(QObject):
    start_workout_signal = Signal(str)  # Signal to trigger workout
    
    def __init__(self, main_window):
        super().__init__()
        self.posture_timer = {
            'start_time': None,
            'notification_sent': False
        }
        self.main_window = main_window
        self.workingOut = False
        # Connect the signal to the main window's slot
        self.start_workout_signal.connect(main_window.start_workout)
    
    def decrement_posture_timer(self):
        """
        Manages a 30-second timer for bad posture notifications.
        - Starts timer when issues are detected
        - Resets timer when all issues are removed
        - Sends notification when timer reaches 0
        """
        
        if( self.workingOut ): return # Don't notify if working out to prevent multiple notifications

        # If timer hasn't started yet, start it
        if self.posture_timer['start_time'] is None:
            self.posture_timer['start_time'] = time()
            self.posture_timer['notification_sent'] = False
            print(f"[TIMER] Bad posture detected, timer started")
        else:
            # Check if 30 seconds have elapsed
            elapsed_time = time() - self.posture_timer['start_time']
            print(f"[TIMER] Elapsed time: {elapsed_time:.1f}s / {TIMER_DURATION}s")
            if elapsed_time >= TIMER_DURATION and not self.posture_timer['notification_sent']:
                print(f"[TIMER] 30 seconds reached! Triggering workout...")
                # Send notification in a separate thread to avoid blocking the camera
                notification_thread = threading.Thread(target=self.send_notification, daemon=True)
                notification_thread.start()
                self.posture_timer['notification_sent'] = True
    
    def reset_timer(self):
        """Reset the timer when good posture is detected."""
        self.posture_timer['start_time'] = None
        self.posture_timer['notification_sent'] = False
    
    def send_notification(self):
        """Send notification in a separate thread to avoid blocking the main camera loop."""
        print("[WORKOUT] Sending notification and starting workout...")
        plyer.notification.notify(
            title='Bad Posture Alert!',
            message='You have been maintaining bad posture for too long. Please correct it. Shrimp'
        )

        self.workingOut = True
        # Emit signal to trigger workout (safe across threads)
        print("[WORKOUT] Emitting start_workout signal...")
        self.start_workout_signal.emit("Back Pain")
    

if __name__ == "__main__":
    # Choose backend like your original main.py comments:
    # Windows: cv2.CAP_DSHOW
    # Mac:     cv2.CAP_AVFOUNDATION
    camera_backend = cv2.CAP_DSHOW

    app = QApplication(sys.argv)

    # FONTS --------------------------------------------------------------------------

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(
        BASE_DIR, "assets", "fonts", "ethereal", "EtherealDemo-SemiBold.otf"
    )

    font_id = QFontDatabase.addApplicationFont(font_path)

    if font_id == -1:
        print("Failed to load font")
        family = "Arial"  # Fallback font
    else:
        families = QFontDatabase.applicationFontFamilies(font_id)
        print("Loaded font families:", families)
        family = families[0]   # THIS is the real name Qt uses
        app.setFont(QFont(family, 10))

    # --------------------------------------------------------------------------------

    win = MainWindow(camera_backend=camera_backend, font_family=family)
    win.show()
    
    sys.exit(app.exec())