"""Skeleton drawing utilities for the unified 12-joint format."""

import cv2
import numpy as np

from src.pose_estimation.keypoint_schema import (
    NUM_UNIFIED_JOINTS,
    UNIFIED_SKELETON,
)

# Colors (BGR) for skeleton segments
SKELETON_COLORS = [
    (0, 255, 0),    # shoulder-shoulder
    (0, 255, 255),  # left shoulder-elbow
    (0, 255, 255),  # left elbow-wrist
    (255, 255, 0),  # right shoulder-elbow
    (255, 255, 0),  # right elbow-wrist
    (0, 128, 255),  # left shoulder-hip
    (255, 0, 128),  # right shoulder-hip
    (0, 255, 0),    # hip-hip
    (0, 128, 255),  # left hip-knee
    (0, 128, 255),  # left knee-ankle
    (255, 0, 128),  # right hip-knee
    (255, 0, 128),  # right knee-ankle
]

# Colors for keypoints (left=cyan, right=magenta, torso=green)
KEYPOINT_COLORS = [
    (0, 255, 255),  # left_shoulder
    (255, 255, 0),  # right_shoulder
    (0, 255, 255),  # left_elbow
    (255, 255, 0),  # right_elbow
    (0, 255, 255),  # left_wrist
    (255, 255, 0),  # right_wrist
    (0, 128, 255),  # left_hip
    (255, 0, 128),  # right_hip
    (0, 128, 255),  # left_knee
    (255, 0, 128),  # right_knee
    (0, 128, 255),  # left_ankle
    (255, 0, 128),  # right_ankle
]


def draw_skeleton(
    frame: np.ndarray,
    unified_kps: np.ndarray,
    confidence_threshold: float = 0.3,
    keypoint_radius: int = 5,
    line_thickness: int = 2,
    label: str | None = None,
) -> np.ndarray:
    """Draw a skeleton overlay using the unified 12-joint format.

    Args:
        frame: BGR image (H, W, 3). Modified in-place.
        unified_kps: (12, 3) array of [x, y, confidence].
        confidence_threshold: Minimum confidence to draw.
        keypoint_radius: Circle radius for joints.
        line_thickness: Thickness of skeleton lines.
        label: Optional text label drawn in top-left corner.

    Returns:
        Frame with skeleton drawn.
    """
    for idx, (start, end) in enumerate(UNIFIED_SKELETON):
        if (unified_kps[start, 2] >= confidence_threshold and
                unified_kps[end, 2] >= confidence_threshold):
            pt1 = (int(unified_kps[start, 0]), int(unified_kps[start, 1]))
            pt2 = (int(unified_kps[end, 0]), int(unified_kps[end, 1]))
            color = SKELETON_COLORS[idx % len(SKELETON_COLORS)]
            cv2.line(frame, pt1, pt2, color, line_thickness, cv2.LINE_AA)

    for idx in range(NUM_UNIFIED_JOINTS):
        if unified_kps[idx, 2] >= confidence_threshold:
            pt = (int(unified_kps[idx, 0]), int(unified_kps[idx, 1]))
            color = KEYPOINT_COLORS[idx]
            cv2.circle(frame, pt, keypoint_radius, color, -1, cv2.LINE_AA)

    if label:
        cv2.putText(
            frame, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return frame


def draw_comparison(
    frame: np.ndarray,
    results: list,
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    """Draw side-by-side skeleton comparisons from different models.

    Args:
        frame: Original BGR frame.
        results: List of PoseResult objects from different models.
        confidence_threshold: Minimum confidence to draw.

    Returns:
        Horizontally concatenated image with one panel per model.
    """
    panels = []
    for result in results:
        panel = frame.copy()
        draw_skeleton(
            panel,
            result.unified_keypoints,
            confidence_threshold=confidence_threshold,
            label=result.model_name,
        )
        panels.append(panel)

    if not panels:
        return frame
    return np.hstack(panels)
