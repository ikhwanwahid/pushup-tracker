"""Self-contained push-up state machine for rep counting.

Includes angle computation and state machine logic so stgcn_study
can run standalone without importing from src/.
"""

import math
from enum import Enum, auto

import numpy as np

# ============================================================
# Joint index constants (unified 12-joint format)
# ============================================================

UNIFIED_JOINT_INDEX = {
    "left_shoulder": 0,
    "right_shoulder": 1,
    "left_elbow": 2,
    "right_elbow": 3,
    "left_wrist": 4,
    "right_wrist": 5,
    "left_hip": 6,
    "right_hip": 7,
    "left_knee": 8,
    "right_knee": 9,
    "left_ankle": 10,
    "right_ankle": 11,
}

# ============================================================
# Angle computation
# ============================================================


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate the angle (degrees) at p2 formed by vectors p1->p2 and p3->p2."""
    v1 = (float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))
    v2 = (float(p3[0]) - float(p2[0]), float(p3[1]) - float(p2[1]))

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])

    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def compute_elbow_angle(unified_kps: np.ndarray, side: str = "best") -> float:
    """Compute elbow angle (shoulder-elbow-wrist).

    Args:
        unified_kps: (12, 3) unified keypoints.
        side: "left", "right", or "best" (highest confidence side).

    Returns:
        Elbow angle in degrees.
    """
    left_joints = ["left_shoulder", "left_elbow", "left_wrist"]
    right_joints = ["right_shoulder", "right_elbow", "right_wrist"]

    left_pts = [unified_kps[UNIFIED_JOINT_INDEX[j]] for j in left_joints]
    right_pts = [unified_kps[UNIFIED_JOINT_INDEX[j]] for j in right_joints]

    if side == "left":
        pts = left_pts
    elif side == "right":
        pts = right_pts
    else:
        left_conf = np.mean([p[2] for p in left_pts])
        right_conf = np.mean([p[2] for p in right_pts])
        pts = left_pts if left_conf >= right_conf else right_pts

    return calculate_angle(pts[0], pts[1], pts[2])


# ============================================================
# State machine
# ============================================================


class PushUpPhase(Enum):
    """Phases of a push-up repetition."""
    UP = auto()
    GOING_DOWN = auto()
    DOWN = auto()
    GOING_UP = auto()


class PushUpStateMachine:
    """Counts push-up reps by tracking elbow angle transitions.

    Features:
        - Moving-average smoothing to reduce noisy keypoint jitter
        - Adaptive calibration: learns the user's actual angle range from the
          first few reps and adjusts thresholds automatically

    State transitions:
        UP -> GOING_DOWN: elbow angle drops below up_threshold
        GOING_DOWN -> DOWN: elbow angle drops below down_threshold
        DOWN -> GOING_UP: elbow angle rises above down_threshold
        GOING_UP -> UP: elbow angle rises above up_threshold (count + 1)
    """

    def __init__(
        self,
        down_threshold: float = 90.0,
        up_threshold: float = 160.0,
        smooth_window: int = 5,
        adaptive: bool = False,
        calibration_reps: int = 2,
    ):
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.state = PushUpPhase.UP
        self.count = 0
        self._frame_idx: int = 0
        self._current_rep_start: int | None = None
        self._rep_boundaries: list[tuple[int, int]] = []

        # Smoothing
        self._smooth_window = smooth_window
        self._angle_buffer: list[float] = []

        # Adaptive calibration
        self._adaptive = adaptive
        self._calibration_reps = calibration_reps
        self._calibrated = False
        self._observed_mins: list[float] = []
        self._observed_maxs: list[float] = []
        self._current_rep_min: float = 180.0
        self._current_rep_max: float = 0.0

    def _smooth(self, raw_angle: float) -> float:
        """Apply moving-average smoothing."""
        self._angle_buffer.append(raw_angle)
        if len(self._angle_buffer) > self._smooth_window:
            self._angle_buffer.pop(0)
        return sum(self._angle_buffer) / len(self._angle_buffer)

    def _recalibrate(self) -> None:
        """Set thresholds based on observed angle range from calibration reps."""
        if not self._observed_mins or not self._observed_maxs:
            return

        avg_min = sum(self._observed_mins) / len(self._observed_mins)
        avg_max = sum(self._observed_maxs) / len(self._observed_maxs)
        angle_range = avg_max - avg_min

        if angle_range < 20:  # Too narrow — don't calibrate
            return

        # Set thresholds at 30% and 70% of observed range
        self.down_threshold = avg_min + angle_range * 0.3
        self.up_threshold = avg_min + angle_range * 0.7
        self._calibrated = True

    def update(self, elbow_angle: float) -> PushUpPhase:
        """Update state machine with a new elbow angle reading."""
        angle = self._smooth(elbow_angle)

        # Track min/max for adaptive calibration
        if self._adaptive and not self._calibrated:
            self._current_rep_min = min(self._current_rep_min, angle)
            self._current_rep_max = max(self._current_rep_max, angle)

        prev_state = self.state

        if self.state == PushUpPhase.UP:
            if angle < self.up_threshold:
                self.state = PushUpPhase.GOING_DOWN

        elif self.state == PushUpPhase.GOING_DOWN:
            if angle <= self.down_threshold:
                self.state = PushUpPhase.DOWN

        elif self.state == PushUpPhase.DOWN:
            if angle > self.down_threshold:
                self.state = PushUpPhase.GOING_UP

        elif self.state == PushUpPhase.GOING_UP:
            if angle >= self.up_threshold:
                self.state = PushUpPhase.UP
                self.count += 1
                if self._current_rep_start is not None:
                    self._rep_boundaries.append(
                        (self._current_rep_start, self._frame_idx)
                    )
                self._current_rep_start = None

                # Adaptive calibration: record this rep's range
                if self._adaptive and not self._calibrated:
                    self._observed_mins.append(self._current_rep_min)
                    self._observed_maxs.append(self._current_rep_max)
                    self._current_rep_min = 180.0
                    self._current_rep_max = 0.0

                    if len(self._observed_mins) >= self._calibration_reps:
                        self._recalibrate()

        # Track rep start: UP -> GOING_DOWN
        if prev_state == PushUpPhase.UP and self.state == PushUpPhase.GOING_DOWN:
            self._current_rep_start = self._frame_idx

        self._frame_idx += 1
        return self.state

    def update_from_keypoints(self, unified_kps: np.ndarray) -> PushUpPhase:
        """Compute elbow angle from keypoints and update."""
        angle = compute_elbow_angle(unified_kps)
        return self.update(angle)

    def reset(self) -> None:
        """Reset counter and state."""
        self.state = PushUpPhase.UP
        self.count = 0
        self._frame_idx = 0
        self._current_rep_start = None
        self._rep_boundaries.clear()
        self._angle_buffer.clear()
        self._observed_mins.clear()
        self._observed_maxs.clear()
        self._current_rep_min = 180.0
        self._current_rep_max = 0.0
        self._calibrated = False

    @property
    def rep_boundaries(self) -> list[tuple[int, int]]:
        """List of (start_frame, end_frame) for each completed rep."""
        return self._rep_boundaries.copy()

    def segment_sequence(self, kps_sequence: np.ndarray) -> list[tuple[int, int]]:
        """Run state machine on a keypoint sequence and return rep boundaries.

        Args:
            kps_sequence: Array of shape (T, 12, 3).

        Returns:
            List of (start_frame, end_frame) tuples, one per completed rep.
        """
        self.reset()
        for kps in kps_sequence:
            self.update_from_keypoints(kps)
        return self.rep_boundaries
