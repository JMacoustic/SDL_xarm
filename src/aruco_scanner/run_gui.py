import sys
from pathlib import Path

import numpy as np
import cv2
from cv2 import aruco

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QPlainTextEdit,
    QSizePolicy,
    QApplication,
    QLabel,
    QPushButton,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QMainWindow,
    QGroupBox,
)

from typing import Optional

from scanner import scan_img_gui
from camera import CamData, load_camera
from mathutils import *


# ── Marker configuration ──────────────────────────────────────────────────────
# Change REF_ID and PLATE_IDS here to use different ArUco marker numbers.
REF_ID    = 1          # fixed reference marker
PLATE_IDS = [2, 3]    # plate markers (must have exactly 2 for the 3-panel GUI)
# ─────────────────────────────────────────────────────────────────────────────

WATCHED_IDS = [REF_ID] + PLATE_IDS   # all IDs tracked by the detector


def _draw_quad(img_bgr: np.ndarray, quad_xy: np.ndarray, color: tuple, thickness: int = 2):
    q = np.asarray(quad_xy, dtype=np.float64).reshape(4, 2)
    pts = q.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for i in range(4):
        p = tuple(np.round(q[i]).astype(int))
        cv2.circle(img_bgr, p, 5, corner_colors[i], -1, lineType=cv2.LINE_AA)


def _draw_axes(img_bgr: np.ndarray, T_obj_cam: Orientation, K: np.ndarray, axis_len: float):
    R = np.asarray(T_obj_cam.rot, dtype=np.float64).reshape(3, 3)
    t = as_col(T_obj_cam.trans)

    O = np.array([[0.0, 0.0, 0.0]], dtype=np.float64).T
    X = np.array([[axis_len, 0.0, 0.0]], dtype=np.float64).T
    Y = np.array([[0.0, axis_len, 0.0]], dtype=np.float64).T
    Z = np.array([[0.0, 0.0, axis_len]], dtype=np.float64).T

    Pw = np.hstack([O, X, Y, Z])
    Pc = (R @ Pw) + t

    PcT = Pc.T
    if np.any(PcT[:, 2] <= 1e-6):
        return

    uv = project_points(PcT, K).astype(np.int32)
    o, px, py, pz = tuple(uv[0]), tuple(uv[1]), tuple(uv[2]), tuple(uv[3])

    cv2.line(img_bgr, o, px, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.line(img_bgr, o, py, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.line(img_bgr, o, pz, (255, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(img_bgr, "x'", px, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img_bgr, "y'", py, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img_bgr, "z'", pz, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)


def _setup_console_box(w, mono: QFont, cols: int, lines: int):
    w.setFont(mono)
    w.setReadOnly(True)
    w.setLineWrapMode(QPlainTextEdit.NoWrap)
    w.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    w.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    w.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    fm = QFontMetrics(mono)
    char_w = fm.horizontalAdvance("M")
    line_h = fm.lineSpacing()

    width  = char_w * cols + 16
    height = line_h * lines + 16
    w.setFixedSize(width, height)


class DetectionThread(QThread):
    # Emits dict[int, MarkerData | None] for all watched IDs
    resultReady = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        frame_bgr: np.ndarray,
        detector: aruco.ArucoDetector,
        camera: CamData,
        len_marker: float,
        watched_ids: list,
    ):
        super().__init__()
        self.frame_bgr  = frame_bgr
        self.detector   = detector
        self.camera     = camera
        self.len_marker = float(len_marker)
        self.watched_ids = watched_ids

    def run(self):
        try:
            markers = scan_img_gui(
                image=self.frame_bgr,
                detector=self.detector,
                camera=self.camera,
                len_marker=self.len_marker,
                watched_ids=self.watched_ids,
            )
            self.resultReady.emit(markers)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, camera: CamData):
        super().__init__()
        self.setWindowTitle("ArUco SE(3) Tracker")

        self.ref_id    = REF_ID
        self.plate_ids = PLATE_IDS

        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        params     = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(dictionary=aruco_dict, detectorParams=params)

        self.cap = cv2.VideoCapture(camera.index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera.index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  camera.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.resolution[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.camera = camera
        self.K = np.asarray(self.camera.camera_mat, dtype=np.float64).reshape(3, 3)

        self.len_marker = 18

        # Zero transforms: plate_id -> T_ref_to_plate_zero
        self.T_zero: dict = {}
        self.T_plate_last: Optional[Orientation] = None

        self.rot_diff_thresh_deg = 20.0

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 480)

        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(9)

        # One text panel per watched ID (ref + 2 plate)
        self.txt_panels: dict = {}
        for mid in WATCHED_IDS:
            w = QPlainTextEdit()
            _setup_console_box(w, mono, cols=44, lines=8)
            w.setPlainText(f"cam-position {mid}:\n-")
            self.txt_panels[mid] = w

        self.txt_rel = QPlainTextEdit()
        _setup_console_box(self.txt_rel, mono, cols=44, lines=18)
        self.txt_rel.setPlainText("relative (after zeroing):\n-")

        self.btn_zero = QPushButton("Zero plate position")
        self.btn_zero.clicked.connect(self.on_zeropos)

        side = QWidget()
        side_layout = QVBoxLayout(side)

        grp1 = QGroupBox("SE(3) Matrices")
        grp1_l = QVBoxLayout(grp1)
        for mid in WATCHED_IDS:
            grp1_l.addWidget(self.txt_panels[mid])

        grp2 = QGroupBox("Relative movement (plate)")
        grp2_l = QVBoxLayout(grp2)
        grp2_l.addWidget(self.btn_zero)
        grp2_l.addWidget(self.txt_rel)

        side_layout.addWidget(grp1)
        side_layout.addWidget(grp2)
        side_layout.addStretch(1)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.addWidget(self.video_label, stretch=4)
        layout.addWidget(side, stretch=1)
        self.setCentralWidget(root)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)

        self.detect_thread = None

        # Last detected marker data keyed by ID
        self.markers_last: dict = {mid: None for mid in WATCHED_IDS}
        # Cached quads and poses for drawing
        self.quads: dict = {mid: None for mid in WATCHED_IDS}
        self.poses: dict = {mid: None for mid in WATCHED_IDS}

    def closeEvent(self, event):
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            if self.detect_thread is not None and self.detect_thread.isRunning():
                self.detect_thread.quit()
                self.detect_thread.wait(200)
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        super().closeEvent(event)

    def _start_detection(self, frame_bgr: np.ndarray):
        if self.detect_thread is not None and self.detect_thread.isRunning():
            return
        self.detect_thread = DetectionThread(
            frame_bgr=frame_bgr.copy(),
            detector=self.detector,
            camera=self.camera,
            len_marker=self.len_marker,
            watched_ids=WATCHED_IDS,
        )
        self.detect_thread.resultReady.connect(self._on_detection_result)
        self.detect_thread.error.connect(self._on_detection_error)
        self.detect_thread.start()

    def _on_detection_error(self, msg: str):
        pass

    def _on_detection_result(self, markers: dict):
        self.markers_last = markers
        for mid, m in markers.items():
            if m is not None:
                self.quads[mid] = np.asarray(m.corner_pos, dtype=np.float64).reshape(4, 2)
                self.poses[mid] = m.orientation
            else:
                self.quads[mid] = None
                self.poses[mid] = None

    def on_zeropos(self):
        try:
            ref_pose = self.poses.get(self.ref_id)
            if ref_pose is None:
                self.T_zero = {}
                self.T_plate_last = None
                self.txt_rel.setPlainText(
                    f"relative (after zero):\nReference marker {self.ref_id} not visible."
                )
                return

            missing_plates = [pid for pid in self.plate_ids if self.poses.get(pid) is None]
            if missing_plates:
                self.T_zero = {}
                self.T_plate_last = None
                self.txt_rel.setPlainText(
                    f"relative (after zero):\nPlate markers not visible: {missing_plates}"
                )
                return

            self.T_zero = {}
            for pid in self.plate_ids:
                T_plate_ref = get_21_transform(ref_pose, self.poses[pid])
                self.T_zero[pid] = inverse_transform(T_plate_ref)

            self.T_plate_last = Orientation(np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))
            self.txt_rel.setPlainText("relative (after zero):\nSaved zero position. Move the plate now.")
        except Exception as e:
            self.T_zero = {}
            self.T_plate_last = None
            self.txt_rel.setPlainText(f"relative (after zero):\nZeroing error:\n{e}")

    def _plate_motion_from(self, pid: int) -> Optional[Orientation]:
        """Compute current plate transform for a single plate marker ID."""
        if not self.T_zero or pid not in self.T_zero:
            return None
        ref_pose   = self.poses.get(self.ref_id)
        plate_pose = self.poses.get(pid)
        if ref_pose is None or plate_pose is None:
            return None
        T_plate_ref = get_21_transform(ref_pose, plate_pose)
        return multiple_transform(T_plate_ref, self.T_zero[pid])

    def update_frame(self):
        try:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return
        except Exception:
            return

        img = frame.copy()
        self._start_detection(frame)

        # Draw each watched marker with a distinct colour
        quad_colors = {
            WATCHED_IDS[0]: (255,   0, 255),
            WATCHED_IDS[1]: (  0, 165, 255),
            WATCHED_IDS[2]: (  0, 255, 128),
        } if len(WATCHED_IDS) >= 3 else {}

        for mid in WATCHED_IDS:
            color = quad_colors.get(mid, (200, 200, 200))
            if self.quads.get(mid) is not None:
                _draw_quad(img, self.quads[mid], color, 2)
            if self.poses.get(mid) is not None:
                _draw_axes(img, self.poses[mid], self.K, axis_len=self.len_marker * 0.5)
            pose = self.poses.get(mid)
            self.txt_panels[mid].setPlainText(
                se3_to_text(pose, f"T_cam_m{mid}") if pose is not None else f"T_cam_m{mid} =\n-"
            )

        if not self.T_zero:
            self.txt_rel.setPlainText(
                f"relative (after zero):\nPress Zero when markers {WATCHED_IDS} are visible."
            )
        else:
            try:
                available = []
                for pid in self.plate_ids:
                    T = self._plate_motion_from(pid)
                    if T is not None:
                        available.append(T)

                Tplate, info = fuse_transforms(available)
                if Tplate is None:
                    self.txt_rel.setPlainText(f"relative (after zeroing):\n{info}")
                else:
                    self.T_plate_last = Tplate
                    yaw_deg = float(angle_from_transform(Tplate) * 180.0 / np.pi)
                    t = as_col(Tplate.trans).reshape(3)
                    self.txt_rel.setPlainText(
                        f"{info}\n\n"
                        "T_plate:\n"
                        f"{se3_to_text(Tplate, 'T_plate')}\n\n"
                        f"yaw(z) [deg]: {yaw_deg:.2f}\n"
                        f"translation: x={t[0]:.4f}, y={t[1]:.4f}, z={t[2]:.4f}"
                    )
            except Exception as e:
                self.txt_rel.setPlainText(f"relative (after zeroing):\nCompute error:\n{e}")

        try:
            rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            pix  = QPixmap.fromImage(qimg)
            pix  = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(pix)
        except Exception:
            pass


def main():
    CAM_NAME     = "frontcam"
    CAM_NPZ_PATH = Path("src") / "camdata" / CAM_NAME / f"{CAM_NAME}.npz"

    try:
        camera = load_camera(CAM_NPZ_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load camera npz: {CAM_NPZ_PATH}\n{e}")

    app = QApplication(sys.argv)
    win = MainWindow(camera=camera)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
