"""Joint angle calculations for push-up analysis."""

import math

import numpy as np

from src.pose_estimation.keypoint_schema import UNIFIED_JOINT_INDEX


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate the angle (degrees) at p2 formed by the vectors p1->p2 and p3->p2.

    Args:
        p1: First point [x, y, ...].
        p2: Vertex point [x, y, ...].
        p3: Third point [x, y, ...].

    Returns:
        Angle in degrees (0-180). Returns 0.0 if any segment has zero length.
    """
    v1 = (float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))
    v2 = (float(p3[0]) - float(p2[0]), float(p3[1]) - float(p2[1]))

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])

    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def _pick_side(unified_kps: np.ndarray, side: str, left_joints: list[str], right_joints: list[str]) -> list[np.ndarray]:
    """Pick left or right side joints, or the side with higher average confidence."""
    left_pts = [unified_kps[UNIFIED_JOINT_INDEX[j]] for j in left_joints]
    right_pts = [unified_kps[UNIFIED_JOINT_INDEX[j]] for j in right_joints]

    if side == "left":
        return left_pts
    elif side == "right":
        return right_pts
    else:  # "best"
        left_conf = np.mean([p[2] for p in left_pts])
        right_conf = np.mean([p[2] for p in right_pts])
        return left_pts if left_conf >= right_conf else right_pts


def compute_elbow_angle(unified_kps: np.ndarray, side: str = "best") -> float:
    """Compute elbow angle (shoulder-elbow-wrist).

    Args:
        unified_kps: (12, 3) unified keypoints.
        side: "left", "right", or "best" (highest confidence side).

    Returns:
        Elbow angle in degrees.
    """
    pts = _pick_side(
        unified_kps, side,
        ["left_shoulder", "left_elbow", "left_wrist"],
        ["right_shoulder", "right_elbow", "right_wrist"],
    )
    return calculate_angle(pts[0], pts[1], pts[2])


def compute_back_alignment(unified_kps: np.ndarray, side: str = "best") -> float:
    """Compute back alignment angle (shoulder-hip-ankle).

    A straight back during push-ups should give ~180 degrees.

    Args:
        unified_kps: (12, 3) unified keypoints.
        side: "left", "right", or "best".

    Returns:
        Back alignment angle in degrees.
    """
    pts = _pick_side(
        unified_kps, side,
        ["left_shoulder", "left_hip", "left_ankle"],
        ["right_shoulder", "right_hip", "right_ankle"],
    )
    return calculate_angle(pts[0], pts[1], pts[2])


def compute_hip_angle(unified_kps: np.ndarray, side: str = "best") -> float:
    """Compute hip angle (shoulder-hip-knee).

    Args:
        unified_kps: (12, 3) unified keypoints.
        side: "left", "right", or "best".

    Returns:
        Hip angle in degrees.
    """
    pts = _pick_side(
        unified_kps, side,
        ["left_shoulder", "left_hip", "left_knee"],
        ["right_shoulder", "right_hip", "right_knee"],
    )
    return calculate_angle(pts[0], pts[1], pts[2])


def compute_knee_angle(unified_kps: np.ndarray, side: str = "best") -> float:
    """Compute knee angle (hip-knee-ankle).

    Args:
        unified_kps: (12, 3) unified keypoints.
        side: "left", "right", or "best".

    Returns:
        Knee angle in degrees.
    """
    pts = _pick_side(
        unified_kps, side,
        ["left_hip", "left_knee", "left_ankle"],
        ["right_hip", "right_knee", "right_ankle"],
    )
    return calculate_angle(pts[0], pts[1], pts[2])


def compute_all_angles(unified_kps: np.ndarray) -> dict[str, float]:
    """Compute all relevant angles for push-up analysis.

    Returns:
        Dictionary with angle names as keys and degrees as values.
    """
    return {
        "elbow_angle": compute_elbow_angle(unified_kps),
        "back_alignment": compute_back_alignment(unified_kps),
        "hip_angle": compute_hip_angle(unified_kps),
        "knee_angle": compute_knee_angle(unified_kps),
    }


def compute_angle_sequence(
    kps_sequence: np.ndarray,
) -> np.ndarray:
    """Compute angles for a sequence of frames.

    Args:
        kps_sequence: Array of shape (T, 12, 3) — unified keypoints per frame.

    Returns:
        Array of shape (T, 4) — [elbow, back, hip, knee] angles per frame.
    """
    T = len(kps_sequence)
    angles = np.zeros((T, 4), dtype=np.float32)
    for t in range(T):
        a = compute_all_angles(kps_sequence[t])
        angles[t, 0] = a["elbow_angle"]
        angles[t, 1] = a["back_alignment"]
        angles[t, 2] = a["hip_angle"]
        angles[t, 3] = a["knee_angle"]
    return angles
