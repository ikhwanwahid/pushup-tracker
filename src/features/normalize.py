"""Torso-length normalization for pose keypoints."""

import numpy as np

from src.pose_estimation.keypoint_schema import UNIFIED_JOINT_INDEX


def torso_normalize(unified_kps: np.ndarray) -> np.ndarray:
    """Normalize keypoints by dividing coordinates by torso length.

    Torso length = average of left and right shoulder-hip distances.
    This makes the representation invariant to subject distance from camera.

    Args:
        unified_kps: (12, 3) unified keypoints [x, y, confidence].

    Returns:
        (12, 3) normalized keypoints. Confidence values are preserved.
        If torso length is zero, returns a copy of the input.
    """
    kps = unified_kps.copy()

    ls = kps[UNIFIED_JOINT_INDEX["left_shoulder"], :2]
    rs = kps[UNIFIED_JOINT_INDEX["right_shoulder"], :2]
    lh = kps[UNIFIED_JOINT_INDEX["left_hip"], :2]
    rh = kps[UNIFIED_JOINT_INDEX["right_hip"], :2]

    left_torso = np.linalg.norm(ls - lh)
    right_torso = np.linalg.norm(rs - rh)
    torso_length = (left_torso + right_torso) / 2.0

    if torso_length < 1e-6:
        return kps

    # Normalize x, y coordinates; keep confidence
    kps[:, :2] /= torso_length
    return kps
