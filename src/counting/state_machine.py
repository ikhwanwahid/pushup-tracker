"""Rule-based push-up counting using a finite state machine."""

from enum import Enum, auto

from src.features.angles import compute_elbow_angle


class PushUpPhase(Enum):
    """Phases of a push-up repetition."""
    UP = auto()
    GOING_DOWN = auto()
    DOWN = auto()
    GOING_UP = auto()


class PushUpStateMachine:
    """Counts push-up repetitions by tracking elbow angle transitions.

    State transitions:
        UP -> GOING_DOWN: elbow angle drops below up_threshold
        GOING_DOWN -> DOWN: elbow angle drops below down_threshold
        DOWN -> GOING_UP: elbow angle rises above down_threshold
        GOING_UP -> UP: elbow angle rises above up_threshold (count + 1)

    Args:
        down_threshold: Elbow angle (degrees) to consider the "down" position.
        up_threshold: Elbow angle (degrees) to consider the "up" position.
    """

    def __init__(
        self,
        down_threshold: float = 90.0,
        up_threshold: float = 160.0,
    ):
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.state = PushUpPhase.UP
        self.count = 0
        self._phase_history: list[PushUpPhase] = []

    def update(self, elbow_angle: float) -> PushUpPhase:
        """Update state machine with a new elbow angle reading.

        Args:
            elbow_angle: Current elbow angle in degrees.

        Returns:
            Current phase after the update.
        """
        if self.state == PushUpPhase.UP:
            if elbow_angle < self.up_threshold:
                self.state = PushUpPhase.GOING_DOWN

        elif self.state == PushUpPhase.GOING_DOWN:
            if elbow_angle <= self.down_threshold:
                self.state = PushUpPhase.DOWN

        elif self.state == PushUpPhase.DOWN:
            if elbow_angle > self.down_threshold:
                self.state = PushUpPhase.GOING_UP

        elif self.state == PushUpPhase.GOING_UP:
            if elbow_angle >= self.up_threshold:
                self.state = PushUpPhase.UP
                self.count += 1

        self._phase_history.append(self.state)
        return self.state

    def update_from_keypoints(self, unified_kps) -> PushUpPhase:
        """Convenience: compute elbow angle from keypoints and update.

        Args:
            unified_kps: (12, 3) unified keypoints.

        Returns:
            Current phase after the update.
        """
        angle = compute_elbow_angle(unified_kps)
        return self.update(angle)

    def reset(self) -> None:
        """Reset counter and state."""
        self.state = PushUpPhase.UP
        self.count = 0
        self._phase_history.clear()

    @property
    def phase_history(self) -> list[PushUpPhase]:
        """Full history of phase labels, one per frame processed."""
        return self._phase_history.copy()

    def label_sequence(self, kps_sequence) -> list[PushUpPhase]:
        """Label an entire keypoint sequence with phases.

        Args:
            kps_sequence: Array of shape (T, 12, 3).

        Returns:
            List of PushUpPhase labels, length T.
        """
        self.reset()
        labels = []
        for kps in kps_sequence:
            labels.append(self.update_from_keypoints(kps))
        return labels
