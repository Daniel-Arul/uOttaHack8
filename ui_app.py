import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import sys
import cv2
import numpy as np
import mediapipe as mp

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QListWidget,
    QFrame, QSpinBox
)

from helpers import extract_pose_data, analyze_posture


# ---------- OpenCV -> Qt ----------
def bgr_to_qimage(frame_bgr):
    h, w, ch = frame_bgr.shape
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


# ---------- Calibration Worker (NO cv2.imshow) ----------
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


# ---------- Posture Worker (NO cv2.imshow) ----------
class PostureWorker(QThread):
    frame_ready = Signal(QImage)
    issues_ready = Signal(list)
    status = Signal(str)
    error = Signal(str)

    def __init__(self, base_data, camera_backend, fps=30, camera_index=0):
        super().__init__()
        self.base_data = base_data
        self.camera_backend = camera_backend
        self.fps = fps
        self.camera_index = camera_index
        self._running = False

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

            self.frame_ready.emit(bgr_to_qimage(frame))
            self.issues_ready.emit(issues)

            self.msleep(int(1000 / max(1, self.fps)))

        cap.release()
        try:
            pose.close()
        except Exception:
            pass

        self.status.emit("Stopped.")


# ---------- Main Window ----------
class MainWindow(QMainWindow):
    def __init__(self, camera_backend):
        super().__init__()
        self.setWindowTitle("Posture Monitor")
        self.resize(1200, 700)

        self.camera_backend = camera_backend
        self.base_data = None

        self.calib_worker = None
        self.posture_worker = None

        # Video preview
        self.video = QLabel("Camera preview")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(860, 560)
        self.video.setObjectName("VideoPanel")

        # Issues list
        self.issues_list = QListWidget()
        self.issues_list.setObjectName("IssuesList")

        # Status line
        self.status_label = QLabel("Calibrate to begin.")
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
        controls.addWidget(QLabel("Calibration:"))
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
        self.setStyleSheet("""
            QMainWindow { background: #0b0f17; color: #e6e6e6; }
            #VideoPanel { background: #111827; border-radius: 18px; }
            #RightPanel { background: #0f172a; border-radius: 18px; padding: 14px; margin-left: 16px; }
            #IssuesList { background: #0b1224; border: 1px solid #22304a; border-radius: 12px; padding: 8px; }
            #StatusLabel { padding: 8px 2px; color: #cbd5e1; }
            QPushButton { background: #1f2937; border: 1px solid #334155; padding: 10px 14px; border-radius: 12px; }
            QPushButton:hover { background: #273449; }
            QPushButton:disabled { opacity: 0.5; }
            QSpinBox { background: #0b1224; border: 1px solid #22304a; border-radius: 10px; padding: 6px; }
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
            self.issues_list.addItem("No issues detected.")
            return
        for issue in issues:
            self.issues_list.addItem(f"{issue['type']}  |  severity: {issue['severity']:.3f}")

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

        self.set_status("Calibration complete. Press Start.")
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
            fps=6,
            camera_index=0
        )
        self.posture_worker.frame_ready.connect(self.on_frame)
        self.posture_worker.issues_ready.connect(self.on_issues)
        self.posture_worker.status.connect(self.set_status)
        self.posture_worker.error.connect(self.on_run_error)
        self.posture_worker.finished.connect(self.on_run_finished)
        self.posture_worker.start()

    def stop_posture(self):
        if self.posture_worker:
            self.posture_worker.stop()
            self.posture_worker.wait()
            self.posture_worker = None

        self.calibrate_btn.setEnabled(True)
        self.start_btn.setEnabled(self.base_data is not None)
        self.stop_btn.setEnabled(False)

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
        event.accept()


if __name__ == "__main__":
    # Choose backend like your original main.py comments:
    # Windows: cv2.CAP_DSHOW
    # Mac:     cv2.CAP_AVFOUNDATION
    camera_backend = cv2.CAP_DSHOW

    app = QApplication(sys.argv)
    win = MainWindow(camera_backend=camera_backend)
    win.show()
    sys.exit(app.exec())