import cv2
import numpy as np
from marker import MarkerData
from mathutils import *
from camera import CamData, load_camera
from pathlib import Path
from typing import Optional


class MixScanner:
    """
    Mixer rotation scanner built on top of the generalised ArUco detection.

    Parameters
    ----------
    cam_name   : camera name whose .npz calibration file is loaded
    len_marker : physical marker side length in mm (or consistent units)
    aruco_dict : name of the cv2.aruco predefined dictionary constant
    ref_id     : ArUco ID of the fixed reference marker (formerly marker 1)
    plate_ids  : list of ArUco IDs that move with the plate (formerly markers 2 & 3)
    """

    def __init__(
        self,
        cam_name: str = "frontcam",
        len_marker: float = 16,
        aruco_dict: str = "DICT_4X4_100",
        ref_id: int = 1,
        plate_ids: Optional[list] = None,
    ):
        CAM_NPZ_PATH = Path("src") / "camdata" / cam_name / f"{cam_name}.npz"
        self.camera = load_camera(CAM_NPZ_PATH)

        aruco_dict_obj = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict))
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary=aruco_dict_obj, detectorParams=parameters)

        self.len_marker = len_marker

        self.ref_id: int = ref_id
        self.plate_ids: list = list(plate_ids) if plate_ids is not None else [2, 3]

        self.obj_corners = np.array([
            [-len_marker / 2.0, -len_marker / 2.0, 0.0],
            [-len_marker / 2.0,  len_marker / 2.0, 0.0],
            [ len_marker / 2.0,  len_marker / 2.0, 0.0],
            [ len_marker / 2.0, -len_marker / 2.0, 0.0],
        ], dtype=np.float64)

        # Detected markers from the last scan, keyed by integer ID.
        # Only ref_id and plate_ids are tracked (set to None when not detected).
        self.markers: dict = {ref_id: None, **{pid: None for pid in self.plate_ids}}

        # Zero-position transforms: plate_id -> T_ref_to_plate_zero
        self.T_zero: dict = {}

        self.T_new_zero: Optional[Orientation] = None
        self.angle: float = 0.0

    def set_camera(self, cam_name: str):
        """Load a different camera calibration by name."""
        CAM_NPZ_PATH = Path("src") / "camdata" / cam_name / f"{cam_name}.npz"
        self.camera = load_camera(CAM_NPZ_PATH)

    def scan_img(self, image: np.ndarray) -> dict:
        """
        Detect ref and plate markers in the image and save as MarkerData.

        Only markers whose IDs are in {ref_id} ∪ plate_ids are stored.
        Undetected markers are set to None.

        Returns
        -------
        dict[int, MarkerData | None] : results for each watched ID
        """
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_corners, ids, _ = self.detector.detectMarkers(img_gray)

        watched = {self.ref_id} | set(self.plate_ids)
        self.markers = {k: None for k in watched}

        if ids is None or len(ids) == 0:
            return self.markers

        for id_arr, img_corner in zip(ids, img_corners):
            marker_id = int(id_arr[0])
            if marker_id not in watched:
                continue

            img_pts = img_corner.reshape(-1, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                self.obj_corners,
                img_pts,
                self.camera.camera_mat,
                self.camera.distortion_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                continue

            R_mat, _ = cv2.Rodrigues(rvec)
            t_vec = tvec.reshape(3, 1)
            self.markers[marker_id] = MarkerData(
                index=marker_id,
                corner_pos=img_pts,
                orientation=Orientation(R_mat, t_vec),
            )

        return self.markers

    def set_zero(self, zero_image: np.ndarray) -> dict:
        """
        Save zero position of the mixer from an image.

        The reference marker and all plate markers must be visible.

        Returns
        -------
        dict[int, Orientation] : zero transforms keyed by plate marker ID
        """
        self.scan_img(zero_image)

        if self.markers.get(self.ref_id) is None:
            raise RuntimeError(
                f"Unable to set zero. Reference marker {self.ref_id} not detected."
            )

        missing = [pid for pid in self.plate_ids if self.markers.get(pid) is None]
        if missing:
            raise RuntimeError(
                f"Unable to set zero. Plate markers not detected: {missing}"
            )

        ref_orient = self.markers[self.ref_id].orientation
        self.T_zero = {}
        for pid in self.plate_ids:
            plate_orient = self.markers[pid].orientation
            T_plate_ref = get_21_transform(ref_orient, plate_orient)
            self.T_zero[pid] = inverse_transform(T_plate_ref)

        self.T_new_zero = Orientation(np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))
        self.angle = 0.0
        self.set_current_transform()

        return self.T_zero

    def set_current_transform(self):
        """
        Calculate current mixer status relative to zero and save.

        Computes a per-plate-marker transform, then fuses all visible estimates.
        """
        if not self.T_zero:
            raise RuntimeError("Zero position not set. Call set_zero() first.")

        if self.markers.get(self.ref_id) is None:
            raise RuntimeError(f"Reference marker {self.ref_id} not detected.")

        ref_orient = self.markers[self.ref_id].orientation
        available = []

        for pid in self.plate_ids:
            if self.markers.get(pid) is None or pid not in self.T_zero:
                continue
            plate_orient = self.markers[pid].orientation
            T_plate_ref = get_21_transform(ref_orient, plate_orient)
            T_plate_new_zero = multiple_transform(T_plate_ref, self.T_zero[pid])
            available.append(T_plate_new_zero)

        if not available:
            raise RuntimeError(
                f"None of the plate markers {self.plate_ids} are currently detected."
            )

        self.T_new_zero, _ = fuse_transforms(available)
        self.angle = angle_from_transform(self.T_new_zero)

    def get_rotation(self, image: np.ndarray) -> float:
        """
        Calculate mixer rotation in degrees from the zero position.

        Parameters
        ----------
        image : BGR numpy image of the rotated mixer

        Returns
        -------
        float : mixer rotation angle in degrees
        """
        self.scan_img(image)
        self.set_current_transform()
        return float(np.degrees(self.angle))

    def get_transform(self, image: np.ndarray) -> Orientation:
        """
        Calculate mixer SE3 transform relative to the zero position.

        Parameters
        ----------
        image : BGR numpy image of the rotated mixer

        Returns
        -------
        Orientation : current mixer transform relative to zero position
        """
        self.scan_img(image)
        self.set_current_transform()
        return self.T_new_zero

    @staticmethod
    def draw_quad(img_bgr: np.ndarray, quad_xy: np.ndarray, color: tuple, thickness: int = 2):
        q = np.asarray(quad_xy, dtype=np.float64).reshape(4, 2)
        pts = q.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        for i in range(4):
            p = tuple(np.round(q[i]).astype(int))
            cv2.circle(img_bgr, p, 5, corner_colors[i], -1, lineType=cv2.LINE_AA)

    @staticmethod
    def draw_axes(img_bgr: np.ndarray, T_obj_cam: Orientation, K: np.ndarray, axis_len: float):
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


# do not use this function directly — only used by run_gui.py.
def scan_img_gui(
    image: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    camera: CamData,
    len_marker: float,
    watched_ids: Optional[list] = None,
) -> dict:
    """
    All-in-one scanner function for GUI development.

    Detects ArUco markers from the given IDs (defaults to [1, 2, 3]) and returns
    a dict[int, MarkerData | None] keyed by each watched ID.

    Parameters
    ----------
    image       : BGR numpy image to scan
    detector    : configured ArucoDetector
    camera      : camera calibration data
    len_marker  : marker side length in the same units as calibration
    watched_ids : which marker IDs to track; defaults to [1, 2, 3]

    Returns
    -------
    dict[int, MarkerData | None]
    """
    if watched_ids is None:
        watched_ids = [1, 2, 3]

    watched = set(watched_ids)
    result: dict = {k: None for k in watched_ids}

    obj_corner = np.array([
        [-len_marker / 2.0, -len_marker / 2.0, 0.0],
        [-len_marker / 2.0,  len_marker / 2.0, 0.0],
        [ len_marker / 2.0,  len_marker / 2.0, 0.0],
        [ len_marker / 2.0, -len_marker / 2.0, 0.0],
    ], dtype=np.float64)

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_corners, ids, _ = detector.detectMarkers(img_gray)

    if ids is None or len(ids) == 0:
        return result

    for id_arr, img_corner in zip(ids, img_corners):
        marker_id = int(id_arr[0])
        if marker_id not in watched:
            continue

        img_pts = img_corner.reshape(-1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj_corner,
            img_pts,
            camera.camera_mat,
            camera.distortion_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        result[marker_id] = MarkerData(
            index=marker_id,
            corner_pos=img_pts,
            orientation=Orientation(R, t),
        )

    return result


if __name__ == "__main__":
    img_new  = cv2.imread("images/test1.jpg")
    img_zero = cv2.imread("images/test2.jpg")

    len_marker = 16

    scanner = MixScanner(
        cam_name="frontcam",
        aruco_dict="DICT_4X4_100",
        len_marker=len_marker,
        ref_id=1,
        plate_ids=[2, 3],
    )

    scanner.set_zero(img_zero)
    angle = scanner.get_rotation(img_new)

    print(angle)
