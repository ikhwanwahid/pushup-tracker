"""Unified 12-joint keypoint format and mapping tables.

The unified format uses the 12 joints most relevant to push-up analysis:
shoulders, elbows, wrists, hips, knees, ankles (left and right).

Each keypoint is stored as [x, y, confidence].
"""

import numpy as np

# Unified joint names and indices
UNIFIED_JOINTS = [
    "left_shoulder",   # 0
    "right_shoulder",  # 1
    "left_elbow",      # 2
    "right_elbow",     # 3
    "left_wrist",      # 4
    "right_wrist",     # 5
    "left_hip",        # 6
    "right_hip",       # 7
    "left_knee",       # 8
    "right_knee",      # 9
    "left_ankle",      # 10
    "right_ankle",     # 11
]

NUM_UNIFIED_JOINTS = len(UNIFIED_JOINTS)

UNIFIED_JOINT_INDEX = {name: i for i, name in enumerate(UNIFIED_JOINTS)}

# Skeleton connections for drawing (indices into unified joints)
UNIFIED_SKELETON = [
    (0, 1),    # shoulder-shoulder
    (0, 2),    # left shoulder-elbow
    (2, 4),    # left elbow-wrist
    (1, 3),    # right shoulder-elbow
    (3, 5),    # right elbow-wrist
    (0, 6),    # left shoulder-hip
    (1, 7),    # right shoulder-hip
    (6, 7),    # hip-hip
    (6, 8),    # left hip-knee
    (8, 10),   # left knee-ankle
    (7, 9),    # right hip-knee
    (9, 11),   # right knee-ankle
]

# ---------------------------------------------------------------------------
# Mapping tables: native index -> unified index
# ---------------------------------------------------------------------------

# YOLO/COCO-17 keypoint order:
# 0:nose 1:left_eye 2:right_eye 3:left_ear 4:right_ear
# 5:left_shoulder 6:right_shoulder 7:left_elbow 8:right_elbow
# 9:left_wrist 10:right_wrist 11:left_hip 12:right_hip
# 13:left_knee 14:right_knee 15:left_ankle 16:right_ankle
YOLO_TO_UNIFIED = {
    5: 0,   # left_shoulder
    6: 1,   # right_shoulder
    7: 2,   # left_elbow
    8: 3,   # right_elbow
    9: 4,   # left_wrist
    10: 5,  # right_wrist
    11: 6,  # left_hip
    12: 7,  # right_hip
    13: 8,  # left_knee
    14: 9,  # right_knee
    15: 10, # left_ankle
    16: 11, # right_ankle
}

# MediaPipe Pose 33 landmarks:
# 11:left_shoulder 12:right_shoulder 13:left_elbow 14:right_elbow
# 15:left_wrist 16:right_wrist 23:left_hip 24:right_hip
# 25:left_knee 26:right_knee 27:left_ankle 28:right_ankle
MEDIAPIPE_TO_UNIFIED = {
    11: 0,  # left_shoulder
    12: 1,  # right_shoulder
    13: 2,  # left_elbow
    14: 3,  # right_elbow
    15: 4,  # left_wrist
    16: 5,  # right_wrist
    23: 6,  # left_hip
    24: 7,  # right_hip
    25: 8,  # left_knee
    26: 9,  # right_knee
    27: 10, # left_ankle
    28: 11, # right_ankle
}

# MoveNet COCO-17 (same layout as YOLO)
MOVENET_TO_UNIFIED = YOLO_TO_UNIFIED.copy()


def map_to_unified(
    native_kps: np.ndarray,
    mapping: dict[int, int],
) -> np.ndarray:
    """Map native keypoints to the unified 12-joint format.

    Args:
        native_kps: Native keypoints, shape (N, 3) where each row is [x, y, confidence].
        mapping: Dict mapping native index -> unified index.

    Returns:
        Array of shape (12, 3) with unified keypoints.
    """
    unified = np.zeros((NUM_UNIFIED_JOINTS, 3), dtype=np.float32)
    for native_idx, unified_idx in mapping.items():
        if native_idx < len(native_kps):
            unified[unified_idx] = native_kps[native_idx]
    return unified
