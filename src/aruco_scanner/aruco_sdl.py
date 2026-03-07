import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from marker import MarkerData
from mathutils import (
    Orientation,
    get_21_transform,
    inverse_transform,
    multiple_transform,
    angle_from_transform,
    fuse_transforms,
)
from camera import CamData, load_camera


class SdlScanner:
    """
    General-purpose ArUco scanner.

    Detects any marker IDs from DICT_4X4_100 and stores them in a dict.
    A single reference marker (ref_id) is treated as the fixed frame; one or
    more plate markers (plate_ids) are tracked relative to it.

    Parameters
    ----------
    cam_name   : camera name whose .npz calibration file is loaded
    len_marker : physical marker side length in mm (or consistent units)
    aruco_dict : name of the cv2.aruco predefined dictionary constant
    ref_id     : ArUco ID of the fixed reference marker
    plate_ids  : list of ArUco IDs that move together on the plate
    """

    def __init__(
        self,
        cam_name: str = "frontcam",
        len_marker: float = 16,
        aruco_dict: str = "DICT_4X4_100",
        ref_id: int = 1,
        plate_ids: Optional[list] = None,
        cam_npz_path: Optional[str] = None,
    ):
        if cam_npz_path is not None:
            npz_path = Path(cam_npz_path)
        else:
            npz_path = Path("src") / "camdata" / cam_name / f"{cam_name}.npz"
        self.camera = load_camera(npz_path)

        aruco_dict_obj = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict))
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary=aruco_dict_obj, detectorParams=parameters)

        self.len_marker = len_marker

        self.obj_corners = np.array([
            [-len_marker / 2.0, -len_marker / 2.0, 0.0],
            [-len_marker / 2.0,  len_marker / 2.0, 0.0],
            [ len_marker / 2.0,  len_marker / 2.0, 0.0],
            [ len_marker / 2.0, -len_marker / 2.0, 0.0],
        ], dtype=np.float64)

        self.ref_id: int = ref_id
        self.plate_ids: list = list(plate_ids) if plate_ids is not None else []

        # All markers detected in the last scan, keyed by integer ID.
        self.markers: dict = {}

        # Zero-position transform for each plate marker.
        # plate_id -> inverse of (plate_id <- ref_id) at zero, i.e. T_ref_to_plate_zero
        self.T_zero: dict = {}

        self.T_new_zero: Optional[Orientation] = None
        self.angle: float = 0.0

    def set_camera(self, cam_name: str = None, cam_npz_path: str = None):
        """Load a different camera calibration by name or by explicit npz path."""
        if cam_npz_path is not None:
            npz_path = Path(cam_npz_path)
        elif cam_name is not None:
            npz_path = Path("src") / "camdata" / cam_name / f"{cam_name}.npz"
        else:
            raise ValueError("Provide either cam_name or cam_npz_path.")
        self.camera = load_camera(npz_path)

    def _solve_marker(self, img_corner: np.ndarray, marker_id: int) -> Optional[MarkerData]:
        """Run solvePnP on a detected corner and return a MarkerData, or None on failure."""
        img_pts = img_corner.reshape(-1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            self.obj_corners,
            img_pts,
            self.camera.camera_mat,
            self.camera.distortion_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        R_mat, _ = cv2.Rodrigues(rvec)
        t_vec = tvec.reshape(3, 1)
        return MarkerData(index=marker_id, corner_pos=img_pts, orientation=Orientation(R_mat, t_vec))

    def scan_img(self, image: np.ndarray) -> dict:
        """
        Detect ALL ArUco markers visible in the image and store as MarkerData.

        There is no limit on which IDs are stored; every successfully solved
        marker is saved in self.markers regardless of whether it appears in
        ref_id or plate_ids.

        Returns
        -------
        dict[int, MarkerData] : detected markers keyed by their integer ID
        """
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_corners, ids, _ = self.detector.detectMarkers(img_gray)

        self.markers = {}

        if ids is None or len(ids) == 0:
            return self.markers

        for id_arr, img_corner in zip(ids, img_corners):
            marker_id = int(id_arr[0])
            marker = self._solve_marker(img_corner, marker_id)
            if marker is not None:
                self.markers[marker_id] = marker

        return self.markers

    def set_zero(self, zero_image: np.ndarray) -> dict:
        """
        Record the zero (reference) position of the plate from an image.

        The reference marker (ref_id) and all plate markers (plate_ids) must
        all be visible in the image.

        Returns
        -------
        dict[int, Orientation] : zero transforms keyed by plate marker ID
        """
        self.scan_img(zero_image)

        if self.ref_id not in self.markers:
            raise RuntimeError(
                f"Reference marker {self.ref_id} not detected in zero image."
            )

        missing = [pid for pid in self.plate_ids if pid not in self.markers]
        if missing:
            raise RuntimeError(
                f"Plate markers not detected in zero image: {missing}. "
                "Make sure all markers are clearly visible."
            )

        ref_orient = self.markers[self.ref_id].orientation
        self.T_zero = {}
        for pid in self.plate_ids:
            plate_orient = self.markers[pid].orientation
            T_plate_ref = get_21_transform(ref_orient, plate_orient)
            self.T_zero[pid] = inverse_transform(T_plate_ref)

        self.T_new_zero = Orientation(np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))
        self.angle = 0.0

        return self.T_zero

    def set_current_transform(self):
        """
        Compute the current plate transform relative to the zero position.

        Uses every visible plate marker to independently estimate the transform,
        then fuses all estimates into self.T_new_zero and self.angle.

        Raises RuntimeError if the reference marker or zero data is missing.
        """
        if not self.T_zero:
            raise RuntimeError("Zero position not set. Call set_zero() first.")

        if self.ref_id not in self.markers:
            raise RuntimeError(f"Reference marker {self.ref_id} not detected.")

        ref_orient = self.markers[self.ref_id].orientation
        available: list = []

        for pid in self.plate_ids:
            if pid not in self.markers or pid not in self.T_zero:
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
        Scan image and return the plate rotation angle in degrees from zero.

        Parameters
        ----------
        image : BGR numpy image of the plate in its current position

        Returns
        -------
        float : rotation angle in degrees
        """
        self.scan_img(image)
        self.set_current_transform()
        return float(np.degrees(self.angle))

    def get_transform(self, image: np.ndarray) -> Orientation:
        """
        Scan image and return the plate SE3 transform relative to zero.

        Parameters
        ----------
        image : BGR numpy image of the plate in its current position

        Returns
        -------
        Orientation : current plate transform relative to zero position
        """
        self.scan_img(image)
        self.set_current_transform()
        return self.T_new_zero


if __name__ == "__main__":
    img_zero = cv2.imread("images/test_zero.jpg")
    img_new  = cv2.imread("images/test_new.jpg")

    scanner = SdlScanner(
        cam_name="frontcam",
        len_marker=16,
        aruco_dict="DICT_4X4_100",
        ref_id=1,
        plate_ids=[2, 3],
    )

    scanner.set_zero(img_zero)
    angle = scanner.get_rotation(img_new)
    print(f"Plate rotation: {angle:.2f} deg")
