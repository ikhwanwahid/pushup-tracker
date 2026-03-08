"""Self-contained push-up state machine for rep counting.

Includes angle computation and state machine logic so r3d_study
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
    """Counts push-up reps using peak/valley detection on the elbow angle.

    Instead of requiring fixed absolute thresholds, this detects reps by
    tracking relative changes in the smoothed elbow angle signal:
      - A "valley" (rep bottom) is detected when the angle drops by at least
        `min_drop` degrees from a recent peak, then starts rising again.
      - A rep completes when the angle recovers by at least `min_rise` degrees
        from the valley.

    This handles bad-form reps that don't reach standard angles (arched back,
    shallow dips, split movements).

    Features:
        - Moving-average smoothing to reduce noisy keypoint jitter
        - Relative peak/valley detection — no fixed angle thresholds needed
        - Tracks rep boundaries (start_frame, end_frame) for classification
    """

    def __init__(
        self,
        smooth_window: int = 5,
        min_drop: float = 30.0,
        min_rise: float = 30.0,
        # Legacy thresholds kept for backward compatibility but unused
        down_threshold: float = 90.0,
        up_threshold: float = 160.0,
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

        # Peak/valley detection
        self._min_drop = min_drop
        self._min_rise = min_rise
        self._recent_peak: float = 0.0  # Highest angle since last valley
        self._recent_valley: float = 180.0  # Lowest angle since last peak
        self._found_valley: bool = False  # Whether we've seen a significant dip

    def _smooth(self, raw_angle: float) -> float:
        """Apply moving-average smoothing."""
        self._angle_buffer.append(raw_angle)
        if len(self._angle_buffer) > self._smooth_window:
            self._angle_buffer.pop(0)
        return sum(self._angle_buffer) / len(self._angle_buffer)

    def update(self, elbow_angle: float) -> PushUpPhase:
        """Update state machine with a new elbow angle reading."""
        angle = self._smooth(elbow_angle)

        # Skip zero angles (person not detected)
        if angle < 1.0:
            self._frame_idx += 1
            return self.state

        prev_state = self.state

        if not self._found_valley:
            # Looking for a valley: track the peak and watch for a drop
            if angle > self._recent_peak:
                self._recent_peak = angle

            self._recent_valley = min(self._recent_valley, angle)
            drop = self._recent_peak - angle

            if drop >= self._min_drop:
                # Significant drop detected — we're in a valley
                self._found_valley = True
                self._recent_valley = angle
                self.state = PushUpPhase.DOWN

                # Mark rep start at the peak before this drop
                if self._current_rep_start is None:
                    self._current_rep_start = max(0, self._frame_idx - 1)
                    self.state = PushUpPhase.GOING_DOWN
        else:
            # In a valley — track the minimum and watch for recovery
            if angle < self._recent_valley:
                self._recent_valley = angle
                self.state = PushUpPhase.DOWN

            rise = angle - self._recent_valley

            if rise >= self._min_rise:
                # Recovery detected — rep complete
                self.state = PushUpPhase.UP
                self.count += 1
                if self._current_rep_start is not None:
                    self._rep_boundaries.append(
                        (self._current_rep_start, self._frame_idx)
                    )
                self._current_rep_start = None

                # Reset for next rep
                self._recent_peak = angle
                self._recent_valley = 180.0
                self._found_valley = False
            elif angle > self._recent_valley:
                self.state = PushUpPhase.GOING_UP

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
        self._recent_peak = 0.0
        self._recent_valley = 180.0
        self._found_valley = False

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
