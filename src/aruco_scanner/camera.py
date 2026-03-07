from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import glob
import numpy as np


@dataclass
class CamData:
    index: int = 0
    name: str = "newcamera"
    resolution: np.ndarray = field(default_factory=lambda: np.zeros(2))
    camera_mat: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    distortion_coeffs: np.ndarray = field(default_factory=lambda: np.zeros((5, 1), dtype=np.float64))


def _reprojection_error(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    K_mat: np.ndarray,
    dist: np.ndarray,
) -> float:
    total_err = 0.0
    total_n = 0
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K_mat, dist)
        proj = proj.reshape(-1, 2)
        imgp2 = imgp.reshape(-1, 2)
        err = np.linalg.norm(imgp2 - proj, axis=1).sum()
        total_err += float(err)
        total_n += imgp2.shape[0]
    return total_err / max(total_n, 1)


def calibrate(
    camera: CamData,
    checker: Tuple[int, int],
    square_size: float = 1.0,
    max_iter: int = 30,
    eps: float = 1e-3,
    show: bool = True,
    image_glob: str = "*.jpg",
) -> CamData:
    """
    Calibrate camera from chessboard images in: src/camdata/<camera.name>/

    Saves:
      src/camdata/<camera.name>/<camera.name>.npz

    Returns:
      camera data

    Notes:
      - checker is the number of squares.
      - square_size scales the 3D object points. Use millimeters
    """
    cols, rows = int(checker[0]-1), int(checker[1]-1)
    if cols <= 1 or rows <= 1:
        raise ValueError("checker must be (cols, rows) of inner corners, both > 1")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(max_iter), float(eps))

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []

    image_dir = Path("src") / "camdata" / str(camera.name)
    images = sorted(glob.glob(str(image_dir / image_glob)))
    if len(images) == 0:
        raise FileNotFoundError(f"No images found: {image_dir / image_glob}")

    last_gray_shape: Optional[Tuple[int, int]] = None

    for fp in images:
        img = cv2.imread(fp)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        last_gray_shape = gray.shape[:2]

        ret, corners = cv2.findChessboardCorners(
            gray,
            (cols, rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK,
        )

        if not ret:
            continue

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp.copy())
        imgpoints.append(corners2)

        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, (cols, rows), corners2, ret)
            cv2.imshow("chessboard", vis)
            cv2.waitKey(200)

    if show:
        cv2.destroyAllWindows()

    if len(objpoints) < 3:
        raise RuntimeError(f"Not enough valid calibration views. Got {len(objpoints)}. Need at least ~3-5.")

    if last_gray_shape is None:
        raise RuntimeError("Failed to read any usable images.")

    h, w = last_gray_shape
    image_size = (w, h)

    ok, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    if not ok:
        raise RuntimeError("cv2.calibrateCamera failed")

    K = K.astype(np.float64)
    dist = dist.astype(np.float64)

    rms = float(ok) if isinstance(ok, (float, np.floating)) else None
    rep = _reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist)

    camera.camera_mat = K
    camera.distortion_coeffs = dist

    image_dir.mkdir(parents=True, exist_ok=True)
    save_path = image_dir / f"{camera.name}.npz"
    np.savez(
        save_path,
        name=str(camera.name),
        index=int(camera.index),
        camera_mat=K,
        distortion_coeffs=dist,
        resolution=np.array(image_size, dtype=np.int32),
        checker=np.array([cols+1, rows+1], dtype=np.int32),
        square_size=np.array([square_size], dtype=np.float64),
        rms=np.array([rms if rms is not None else np.nan], dtype=np.float64),
        mean_reprojection_error=np.array([rep], dtype=np.float64),
        n_images=np.array([len(images)], dtype=np.int32),
        n_valid=np.array([len(objpoints)], dtype=np.int32),
    )

    print(f"[calibrate] saved: {save_path}")
    if rms is not None:
        print(f"[calibrate] RMS (OpenCV): {rms:.6f}")
    print(f"[calibrate] mean reprojection error: {rep:.6f} px")
    print(f"[calibrate] K=\n{K}")
    print(f"[calibrate] dist=\n{dist.reshape(-1)}")

    return camera

def load_camera(npz_path: str | Path) -> CamData:
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    name = str(data.get("name", npz_path.stem))
    index = int(data.get("index", 0))

    resolution = np.asarray(data["resolution"], dtype=np.int32)

    K = np.asarray(data["camera_mat"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(data["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)

    return CamData(index=index, name=name, resolution=resolution, camera_mat=K, distortion_coeffs=dist)

if __name__ == "__main__":
    cam = CamData(index=1, name="phonecam")
    calibrate(
        camera=cam,
        checker=(8, 7),
        square_size=16,   # set to your real square size in millimeter
        show=True,
        image_glob="*.jpg",
    )
