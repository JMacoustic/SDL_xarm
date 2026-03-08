#!/usr/bin/env python3
"""
ArUco-guided pick and place node for xArm7 with G2 gripper.

Two ArUco markers (DICT_4X4_100) are used:
  - Floor reference marker (FLOOR_REF_ID): rigidly fixed to the ground/table.
    Its SE3 pose in the robot base frame is configured below as T_BASE_REF.
  - Object marker (OBJECT_ID): attached to the object to be picked up.

Coordinate chain:
    T_base_obj = T_BASE_REF @ inv(T_cam_ref) @ T_cam_obj

where T_cam_* are the raw solvePnP poses (camera←marker) returned by the
ArUco scanner.  The gripper approaches along the Z-axis of T_base_obj,
so the gripper XY plane is parallel to the object marker's XY plane.

Usage (run in separate terminals):
  Terminal 1 — start planner stack:
    ros2 launch testapp base_realmove.launch.py robot_ip:=<IP>
  Terminal 2 — run this node:
    ros2 run testapp aruco_pick_and_place
"""

import math
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import PlanSingleStraight, PlanPose, PlanJoint, PlanExec, SetInt16

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QPlainTextEdit,
    QSizePolicy, QVBoxLayout, QWidget,
)

# ── Import the ArUco scanner from its sibling package ───────────────────────
_ARUCO_SCANNER_DIR = Path('/home/juhyeon/sdl_ws/src/aruco_scanner')
sys.path.insert(0, str(_ARUCO_SCANNER_DIR))
from aruco_sdl import SdlScanner          # noqa: E402
from mathutils import project_points      # noqa: E402


# ===========================================================================
#  USER CONFIGURATION  ← edit this block to match your setup
# ===========================================================================

# -- Camera ------------------------------------------------------------------
CAMERA_INDEX = 0
CAM_NAME     = "frontcam"
MARKER_LEN_MM = 50.0

# -- ArUco marker IDs --------------------------------------------------------
FLOOR_REF_ID = 10   # fixed floor reference marker
OBJECT_ID    = 11   # marker on the object to pick up

# -- T_BASE_REF: pose of the FLOOR reference marker in robot base frame ------
# p_base = T_BASE_REF @ p_marker_hom   (translation in metres)
T_BASE_REF: np.ndarray = np.array([
    [ 1.0,  0.0,  0.0,  0.50],
    [ 0.0,  1.0,  0.0, -0.10],
    [ 0.0,  0.0,  1.0,  0.00],
    [ 0.0,  0.0,  0.0,  1.00],
], dtype=np.float64)

# -- Destination (robot base frame) ------------------------------------------
DEST_SE3: np.ndarray = np.array([
    [ 1.0,  0.0,  0.0,  0.40],
    [ 0.0,  1.0,  0.0,  0.20],
    [ 0.0,  0.0,  1.0,  0.20],
    [ 0.0,  0.0,  0.0,  1.00],
], dtype=np.float64)

# -- Robot motion parameters -------------------------------------------------
HOME_JOINTS        = [math.radians(a) for a in [0, -45, 0, 45, 0, 90, 0]]
APPROACH_DIST      = 0.10    # hover distance (m)
GRIPPER_OPEN       = 0.85
GRIPPER_CLOSED     = 0.00
COLLISION_SENSITIVITY = 3


# ===========================================================================
#  SE3 / geometry helpers
# ===========================================================================

def _rotation_to_quaternion(R: np.ndarray):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def _se3_to_pose(T: np.ndarray) -> Pose:
    pose = Pose()
    pose.position.x = float(T[0, 3])
    pose.position.y = float(T[1, 3])
    pose.position.z = float(T[2, 3])
    qx, qy, qz, qw = _rotation_to_quaternion(T[:3, :3])
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def _approach_pose(T: np.ndarray, dist: float) -> Pose:
    z_axis = T[:3, 2]
    T_hover = T.copy()
    T_hover[:3, 3] -= dist * z_axis
    return _se3_to_pose(T_hover)


def _inv_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3]  = -(R.T @ t)
    return T_inv


def _orientation_to_se3(orient) -> np.ndarray:
    """Orientation (rot mm-scale) → 4×4 SE3 with translation in metres."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = orient.rot
    T[:3, 3]  = orient.trans.reshape(3) / 1000.0
    return T


# ===========================================================================
#  Drawing helpers (used by _CameraWorker)
# ===========================================================================

def _draw_quad(img: np.ndarray, quad_xy: np.ndarray,
               color: tuple, thickness: int = 2) -> None:
    q = np.asarray(quad_xy, dtype=np.float64).reshape(4, 2)
    pts = q.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)
    corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for i in range(4):
        p = tuple(np.round(q[i]).astype(int))
        cv2.circle(img, p, 5, corner_colors[i], -1, lineType=cv2.LINE_AA)


def _draw_axes(img: np.ndarray, orient, K: np.ndarray, axis_len: float) -> None:
    R = np.asarray(orient.rot,   dtype=np.float64).reshape(3, 3)
    t = np.asarray(orient.trans, dtype=np.float64).reshape(3, 1)

    pts_obj = np.array([
        [0.0,      0.0,      0.0     ],
        [axis_len, 0.0,      0.0     ],
        [0.0,      axis_len, 0.0     ],
        [0.0,      0.0,      axis_len],
    ], dtype=np.float64)

    Pc = (R @ pts_obj.T) + t
    if np.any(Pc[2, :] <= 1e-6):
        return

    uv = project_points(Pc.T, K).astype(np.int32)
    o, px, py, pz = tuple(uv[0]), tuple(uv[1]), tuple(uv[2]), tuple(uv[3])

    cv2.line(img, o, px, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.line(img, o, py, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.line(img, o, pz, (255, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(img, "x'", px, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "y'", py, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "z'", pz, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)


# ===========================================================================
#  Shared state (thread-safe buffer between camera worker and ROS node)
# ===========================================================================

class _SharedState:
    """Thread-safe store for the latest camera frame, marker detections, and phase text."""

    def __init__(self):
        self._lock   = threading.Lock()
        self._frame  : np.ndarray | None = None
        self._markers: dict               = {}
        self._status : str                = 'Starting…'

    def set_frame(self, frame: np.ndarray, markers: dict) -> None:
        with self._lock:
            self._frame   = frame
            self._markers = markers

    def get_frame(self):
        with self._lock:
            return self._frame, dict(self._markers)

    def set_status(self, text: str) -> None:
        with self._lock:
            self._status = text

    def get_status(self) -> str:
        with self._lock:
            return self._status


# ===========================================================================
#  Camera worker — owns VideoCapture + SdlScanner, runs in its own QThread
# ===========================================================================

class _CameraWorker(QThread):
    """Continuously captures frames, runs ArUco detection, draws overlays."""

    frameReady = Signal(object)   # emits annotated BGR np.ndarray

    # Quad colours per role
    _COLORS = {
        'ref': (255,   0, 255),   # magenta  — floor reference
        'obj': (  0, 165, 255),   # orange   — object
        'unk': (200, 200, 200),   # grey     — any other detected marker
    }

    def __init__(self, shared: _SharedState, scanner: SdlScanner):
        super().__init__()
        self._shared  = shared
        self._scanner = scanner
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            self._shared.set_status(f'ERROR: cannot open camera {CAMERA_INDEX}')
            return

        K = np.asarray(self._scanner.camera.camera_mat, dtype=np.float64)

        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            markers = self._scanner.scan_img(frame)

            vis = frame.copy()
            for mid, m in markers.items():
                if m is None:
                    continue
                if mid == FLOOR_REF_ID:
                    color = self._COLORS['ref']
                elif mid == OBJECT_ID:
                    color = self._COLORS['obj']
                else:
                    color = self._COLORS['unk']
                _draw_quad(vis, m.corner_pos, color)
                _draw_axes(vis, m.orientation, K, MARKER_LEN_MM * 0.5)

            self._shared.set_frame(vis, markers)
            self.frameReady.emit(vis)

        cap.release()


# ===========================================================================
#  GUI window
# ===========================================================================

class _CameraWindow(QMainWindow):
    """Streams the annotated camera feed and shows marker / phase status."""

    def __init__(self, shared: _SharedState, worker: _CameraWorker):
        super().__init__()
        self.setWindowTitle('ArUco Pick and Place')
        self._shared = shared

        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setMinimumSize(640, 480)
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        mono = QFont('Consolas')
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(10)

        self._status_box = QPlainTextEdit()
        self._status_box.setFont(mono)
        self._status_box.setReadOnly(True)
        self._status_box.setFixedHeight(80)
        self._status_box.setPlainText('Waiting for camera…')

        layout = QVBoxLayout()
        layout.addWidget(self._video_label)
        layout.addWidget(self._status_box)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)

        worker.frameReady.connect(self._on_frame)

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(200)

    def _on_frame(self, frame: np.ndarray) -> None:
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        pix  = pix.scaled(self._video_label.size(),
                           Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._video_label.setPixmap(pix)

    def _refresh_status(self) -> None:
        _, markers = self._shared.get_frame()
        phase      = self._shared.get_status()

        def _det(mid):
            return 'detected' if (markers.get(mid) is not None) else 'not visible'

        lines = [
            f'Ref [{FLOOR_REF_ID}]: {_det(FLOOR_REF_ID)}   '
            f'Obj [{OBJECT_ID}]: {_det(OBJECT_ID)}',
            f'Phase: {phase}',
        ]
        self._status_box.setPlainText('\n'.join(lines))


# ===========================================================================
#  ROS worker — runs node.run() in a QThread so Qt main thread stays free
# ===========================================================================

class _RosWorker(QThread):
    def __init__(self, node):
        super().__init__()
        self._node = node

    def run(self):
        try:
            self._node.run()
        except Exception as e:
            self._node.get_logger().error(f'Pick-and-place error: {e}')


# ===========================================================================
#  ROS2 node
# ===========================================================================

class ArucoPickAndPlaceNode(Node):
    def __init__(self, shared: _SharedState):
        super().__init__('aruco_pick_and_place')
        self._shared = shared

        # ── Arm planner clients ──────────────────────────────────────────────
        self._straight_plan_client = self.create_client(
            PlanSingleStraight, 'xarm_straight_plan')
        self._pose_plan_client     = self.create_client(PlanPose,  'xarm_pose_plan')
        self._joint_plan_client    = self.create_client(PlanJoint, 'xarm_joint_plan')
        self._exec_client          = self.create_client(PlanExec,  'xarm_exec_plan')

        # ── Gripper clients ──────────────────────────────────────────────────
        self._gripper_plan_client  = self.create_client(
            PlanJoint, 'xarm_gripper_joint_plan')
        self._gripper_exec_client  = self.create_client(
            PlanExec,  'xarm_gripper_exec_plan')

        # ── Collision sensitivity ────────────────────────────────────────────
        self._collision_client = self.create_client(
            SetInt16, '/xarm/set_collision_sensitivity')

        self.get_logger().info('Waiting for arm planner services…')
        self._straight_plan_client.wait_for_service()
        self._pose_plan_client.wait_for_service()
        self._joint_plan_client.wait_for_service()
        self._exec_client.wait_for_service()
        self.get_logger().info('Waiting for gripper services…')
        self._gripper_plan_client.wait_for_service()
        self._gripper_exec_client.wait_for_service()
        self.get_logger().info('All services ready.')

        # ── Robot error monitoring ───────────────────────────────────────────
        self._robot_error: int = 0
        self.create_subscription(
            RobotMsg, '/xarm/robot_states', self._robot_state_cb, 10)

        if self._collision_client.wait_for_service(timeout_sec=2.0):
            req = SetInt16.Request()
            req.data = COLLISION_SENSITIVITY
            self._collision_client.call_async(req)
            self.get_logger().info(
                f'Collision sensitivity set to {COLLISION_SENSITIVITY}/5')

        # ── ArUco scanner (owned here; shared with _CameraWorker) ────────────
        cam_npz = str(
            _ARUCO_SCANNER_DIR / "camdata" / CAM_NAME / f"{CAM_NAME}.npz"
        )
        self.scanner = SdlScanner(
            cam_name=CAM_NAME,
            len_marker=MARKER_LEN_MM,
            aruco_dict='DICT_4X4_100',
            ref_id=FLOOR_REF_ID,
            plate_ids=[OBJECT_ID],
            cam_npz_path=cam_npz,
        )

    # ── Robot error callback ─────────────────────────────────────────────────

    def _robot_state_cb(self, msg: RobotMsg) -> None:
        if msg.err != 0 and self._robot_error == 0:
            self.get_logger().error(f'Robot error (code {msg.err}) — stopping.')
        self._robot_error = msg.err

    def _check_error(self) -> bool:
        if self._robot_error != 0:
            self.get_logger().error('Robot error active, aborting.')
            return False
        return True

    # ── Arm motion helpers ───────────────────────────────────────────────────

    def _plan_joints(self, angles_rad: list) -> bool:
        req = PlanJoint.Request()
        req.target = [float(a) for a in angles_rad]
        future = self._joint_plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error('xarm_joint_plan: no response')
            return False
        return bool(future.result().success)

    def _plan_pose(self, pose: Pose) -> bool:
        req = PlanPose.Request()
        req.target = pose
        future = self._pose_plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error('xarm_pose_plan: no response')
            return False
        return bool(future.result().success)

    def _plan_straight(self, pose: Pose) -> bool:
        req = PlanSingleStraight.Request()
        req.target = pose
        future = self._straight_plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error('xarm_straight_plan: no response')
            return False
        return bool(future.result().success)

    def _execute(self) -> bool:
        req = PlanExec.Request()
        req.wait = True
        future = self._exec_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error('xarm_exec_plan: no response')
            return False
        return bool(future.result().success)

    def _move_to_joints(self, angles: list, label: str) -> bool:
        self.get_logger().info(f'Moving to {label} (joint-space)…')
        if not self._check_error():       return False
        if not self._plan_joints(angles): return False
        return self._execute()

    def _move_to_pose(self, pose: Pose, label: str) -> bool:
        self.get_logger().info(f'Moving to {label} (OMPL)…')
        if not self._check_error():   return False
        if not self._plan_pose(pose): return False
        return self._execute()

    def _move_straight_to(self, pose: Pose, label: str) -> bool:
        self.get_logger().info(f'Straight move to {label}…')
        if not self._check_error():       return False
        if not self._plan_straight(pose): return False
        return self._execute()

    # ── Gripper helpers ──────────────────────────────────────────────────────

    def _move_gripper(self, position: float) -> bool:
        req = PlanJoint.Request()
        req.target = [float(position)]
        future = self._gripper_plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None or not future.result().success:
            self.get_logger().error('Gripper plan failed')
            return False
        exec_req = PlanExec.Request()
        exec_req.wait = True
        future = self._gripper_exec_client.call_async(exec_req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None or not future.result().success:
            self.get_logger().error('Gripper execute failed')
            return False
        return True

    def _open_gripper(self) -> bool:
        self.get_logger().info('Opening gripper…')
        return self._move_gripper(GRIPPER_OPEN)

    def _close_gripper(self) -> bool:
        self.get_logger().info('Closing gripper…')
        return self._move_gripper(GRIPPER_CLOSED)

    # ── ArUco pose detection ─────────────────────────────────────────────────

    def _scan_object_pose(self) -> np.ndarray:
        """
        Poll the shared camera state until both markers are visible.

        Returns 4×4 SE3 of the object in robot base frame:
            T_base_obj = T_BASE_REF @ inv(T_cam_ref) @ T_cam_obj

        Blocks indefinitely (no timeout). Close the window or Ctrl-C to abort.
        """
        self.get_logger().info(
            f'Waiting for markers: ref={FLOOR_REF_ID}, object={OBJECT_ID}…'
        )
        while rclpy.ok():
            _, markers = self._shared.get_frame()

            ref_marker = markers.get(FLOOR_REF_ID)
            obj_marker = markers.get(OBJECT_ID)

            if ref_marker is None:
                self.get_logger().info(
                    f'Waiting for floor reference marker {FLOOR_REF_ID}…',
                    throttle_duration_sec=2.0)
                time.sleep(0.02)
                continue

            if obj_marker is None:
                self.get_logger().info(
                    f'Waiting for object marker {OBJECT_ID}…',
                    throttle_duration_sec=2.0)
                time.sleep(0.02)
                continue

            T_cam_ref  = _orientation_to_se3(ref_marker.orientation)
            T_cam_obj  = _orientation_to_se3(obj_marker.orientation)
            T_base_obj = T_BASE_REF @ _inv_se3(T_cam_ref) @ T_cam_obj

            p = T_base_obj[:3, 3]
            self.get_logger().info(
                f'Object in base frame: x={p[0]:.3f}  y={p[1]:.3f}  z={p[2]:.3f} m'
            )
            return T_base_obj

        raise RuntimeError('ROS2 shut down while waiting for markers.')

    # ── Pick-and-place sequence ───────────────────────────────────────────────

    def run(self):
        """
        Full ArUco-guided pick-and-place cycle (12 steps).
        Phase text is written to _SharedState so the GUI can display it.
        """
        def phase(text: str):
            self._shared.set_status(text)
            self.get_logger().info(text)

        phase('Scanning for object…')
        grab_se3 = self._scan_object_pose()

        pre_grasp = _approach_pose(grab_se3, APPROACH_DIST)
        grasp     = _se3_to_pose(grab_se3)
        pre_place = _approach_pose(DEST_SE3,  APPROACH_DIST)
        place     = _se3_to_pose(DEST_SE3)

        phase('=== Pick and Place Start ===')

        phase('Moving to home…')
        if not self._move_to_joints(HOME_JOINTS, 'home'):         return

        phase('Opening gripper…')
        if not self._open_gripper():                               return

        phase('Pre-grasp hover (OMPL)…')
        if not self._move_to_pose(pre_grasp, 'pre-grasp hover'):  return

        phase('Approaching grasp…')
        if not self._move_straight_to(grasp, 'grasp'):            return

        phase('Closing gripper…')
        if not self._close_gripper():                              return

        phase('Departing grasp…')
        if not self._move_straight_to(pre_grasp, 'post-grasp'):   return

        phase('Pre-place hover (OMPL)…')
        if not self._move_to_pose(pre_place, 'pre-place hover'):  return

        phase('Descending to place…')
        if not self._move_straight_to(place, 'place'):            return

        phase('Opening gripper (release)…')
        if not self._open_gripper():                               return

        phase('Departing place…')
        if not self._move_straight_to(pre_place, 'post-place'):   return

        phase('Returning home…')
        if not self._move_to_joints(HOME_JOINTS, 'home'):         return

        phase('=== Complete ===')


# ===========================================================================
#  Entry point
# ===========================================================================

def main(args=None):
    rclpy.init(args=args)

    app = QApplication(sys.argv)

    shared = _SharedState()
    node   = ArucoPickAndPlaceNode(shared)

    cam_worker = _CameraWorker(shared, node.scanner)
    cam_worker.start()

    window = _CameraWindow(shared, cam_worker)
    window.show()

    ros_worker = _RosWorker(node)
    ros_worker.start()

    try:
        app.exec()
    except KeyboardInterrupt:
        pass
    finally:
        cam_worker.stop()
        cam_worker.wait()
        ros_worker.wait()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
