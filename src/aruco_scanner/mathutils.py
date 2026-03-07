import numpy as np
from typing import Optional, Tuple

class Orientation:
    def __init__(self, rotation: np.ndarray = np.identity(3), translation: np.ndarray = np.zeros((3, 1))):
        self.rot = rotation
        self.trans = translation
    
    def reset(self):
        self.rot = np.identity(3)
        self.trans = np.zeros((3, 1))

def inverse_transform(T: Orientation):
    R_mat = T.rot
    P_vec = T.trans

    R_inv = R_mat.T
    P_inv = - np.matmul(R_inv, P_vec)

    return Orientation(R_inv, P_inv)


def multiple_transform(T1: Orientation, T2: Orientation):
    R_mat1 = T1.rot
    P_vec1 = T1.trans
    R_mat2 = T2.rot
    P_vec2 = T2.trans

    R_mul = np.matmul(R_mat1, R_mat2)
    P_mul = np.matmul(R_mat1, P_vec2) + P_vec1

    return Orientation(R_mul, P_mul)

def angle_from_transform(T: Orientation):
    R_mat = T.rot
    theta = np.arctan2(R_mat[1, 0], R_mat[0, 0])
    return theta


def get_21_transform(transform_c1: Orientation, transform_c2: Orientation) -> Orientation:
    """Input 2 SE3 transforms that shares same reference frame. Returns relative SE3 transform between them"""
    T_2c = inverse_transform(transform_c2)
    T_c1 = transform_c1

    transform_21 = multiple_transform(T_2c, T_c1)

    return transform_21

def get_camera_intrinsic(
    fx: float, fy: float,
    cx: float, cy: float
) -> np.ndarray:
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    return K

def as_col(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.shape == (3,):
        return v.reshape(3, 1)
    if v.shape == (3, 1):
        return v
    v = v.reshape(-1)
    if v.size == 3:
        return v.reshape(3, 1)
    return np.zeros((3, 1), dtype=np.float64)

def se3_to_text(T: Orientation, name: str = "T") -> str:
    R = np.asarray(T.rot, dtype=np.float64).reshape(3, 3)
    t = as_col(T.trans)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3:4] = t
    return f"{name} =\n{np.array2string(M, precision=4, suppress_small=True)}"

def project_points(Xc: np.ndarray, K: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    Xc = np.asarray(Xc, dtype=np.float64)
    z = Xc[:, 2:3]
    z = np.where(np.abs(z) < eps, eps, z)
    u = K[0, 0] * (Xc[:, 0:1] / z) + K[0, 2]
    v = K[1, 1] * (Xc[:, 1:2] / z) + K[1, 2]
    return np.hstack([u, v])


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = (q / n).tolist()

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-12)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-12)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = _clamp(dot, -1.0, 1.0)

    if dot > 0.9995:
        q = q0 + float(t) * (q1 - q0)
        return q / max(float(np.linalg.norm(q)), 1e-12)

    theta0 = np.arccos(dot)
    sin_theta0 = np.sin(theta0)
    theta = theta0 * float(t)
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * (sin_theta / sin_theta0)
    s1 = sin_theta / sin_theta0
    q = (s0 * q0) + (s1 * q1)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    Ra = np.asarray(Ra, dtype=np.float64).reshape(3, 3)
    Rb = np.asarray(Rb, dtype=np.float64).reshape(3, 3)
    R = Ra.T @ Rb
    c = (float(np.trace(R)) - 1.0) * 0.5
    c = _clamp(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def fuse_transforms(transforms: list) -> Tuple[Optional[Orientation], str]:
    """Fuse any number of SE3 transforms by averaging quaternions and translations."""
    valid = [T for T in transforms if T is not None]
    if not valid:
        return None, "No transforms to fuse."
    if len(valid) == 1:
        return valid[0], "Single transform."

    quats = np.array([rot_to_quat(T.rot) for T in valid], dtype=np.float64)
    # Ensure quaternion sign consistency (flip if dot with first is negative)
    for i in range(1, len(quats)):
        if np.dot(quats[0], quats[i]) < 0.0:
            quats[i] = -quats[i]
    q_avg = np.mean(quats, axis=0)
    norm = float(np.linalg.norm(q_avg))
    q_avg = q_avg / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])
    R_avg = quat_to_rot(q_avg)

    t_avg = np.mean([as_col(T.trans) for T in valid], axis=0)

    return Orientation(R_avg, t_avg), f"Fused {len(valid)} transforms."


def pick_or_fuse(T22: Optional[Orientation], T33: Optional[Orientation]) -> Tuple[Optional[Orientation], str]:
    if T22 is None and T33 is None:
        return None, "Need (1&2) or (1&3)."
    if T22 is not None and T33 is None:
        return T22, "Using pair 1-2."
    if T33 is not None and T22 is None:
        return T33, "Using pair 1-3."

    R22, t22 = np.asarray(T22.rot, dtype=np.float64).reshape(3, 3), as_col(T22.trans)
    R33, t33 = np.asarray(T33.rot, dtype=np.float64).reshape(3, 3), as_col(T33.trans)

    drot = rot_angle_deg(R22, R33)

    q22 = rot_to_quat(R22)
    q33 = rot_to_quat(R33)
    qavg = quat_slerp(q22, q33, 0.5)
    Ravg = quat_to_rot(qavg)
    tavg = 0.5 * (t22 + t33)
    return Orientation(Ravg, tavg), f"Fused 2 & 3. diff: {drot:.1f}deg"
