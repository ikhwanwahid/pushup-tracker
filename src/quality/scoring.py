"""Per-criterion quality scoring for push-up repetitions."""

from dataclasses import dataclass

import numpy as np

from src.features.angles import compute_angle_sequence


@dataclass
class RepScore:
    """Quality score for a single push-up repetition.

    All scores are 0-100 (higher is better).
    """
    back_alignment: float
    depth: float
    extension: float
    composite: float


class QualityScorer:
    """Scores individual push-up reps from angle sequences.

    Args:
        weight_back: Weight for back alignment in composite score.
        weight_depth: Weight for depth in composite score.
        weight_extension: Weight for extension in composite score.
    """

    def __init__(
        self,
        weight_back: float = 0.4,
        weight_depth: float = 0.35,
        weight_extension: float = 0.25,
    ):
        self.weight_back = weight_back
        self.weight_depth = weight_depth
        self.weight_extension = weight_extension

    def score_rep(self, rep_kps: np.ndarray) -> RepScore:
        """Score a single rep from its keypoint sequence.

        Args:
            rep_kps: (T, 12, 3) unified keypoints for one rep.

        Returns:
            RepScore with per-criterion and composite scores.
        """
        angles = compute_angle_sequence(rep_kps)
        # angles columns: [elbow, back, hip, knee]

        back_score = self._score_back_alignment(angles[:, 1])
        depth_score = self._score_depth(angles[:, 0])
        extension_score = self._score_extension(angles[:, 0])

        composite = (
            self.weight_back * back_score
            + self.weight_depth * depth_score
            + self.weight_extension * extension_score
        )

        return RepScore(
            back_alignment=back_score,
            depth=depth_score,
            extension=extension_score,
            composite=composite,
        )

    @staticmethod
    def _score_back_alignment(back_angles: np.ndarray) -> float:
        """Score back alignment: ideal is ~180 degrees throughout.

        Penalizes deviation from 180 (sagging or piking).
        """
        mean_deviation = np.mean(np.abs(back_angles - 180.0))
        # 0 deviation = 100, 45+ deviation = 0
        score = max(0.0, 100.0 - (mean_deviation / 45.0) * 100.0)
        return float(score)

    @staticmethod
    def _score_depth(elbow_angles: np.ndarray) -> float:
        """Score depth: minimum elbow angle should reach ~90 degrees.

        Lower minimum angle = better depth.
        """
        min_angle = float(np.min(elbow_angles))
        # 90 degrees or less = 100, 160+ = 0
        if min_angle <= 90:
            return 100.0
        score = max(0.0, 100.0 - ((min_angle - 90.0) / 70.0) * 100.0)
        return float(score)

    @staticmethod
    def _score_extension(elbow_angles: np.ndarray) -> float:
        """Score extension: maximum elbow angle should reach ~170 degrees.

        Higher max angle = better extension.
        """
        max_angle = float(np.max(elbow_angles))
        # 170+ degrees = 100, 120 or less = 0
        if max_angle >= 170:
            return 100.0
        score = max(0.0, ((max_angle - 120.0) / 50.0) * 100.0)
        return float(score)
