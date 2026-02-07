"""Tests for push-up quality scoring."""

import numpy as np

from src.quality.scoring import QualityScorer, RepScore
from src.quality.feedback import generate_feedback


def _make_rep_kps(elbow_min: float, elbow_max: float, back_angle: float, n_frames: int = 30) -> np.ndarray:
    """Create synthetic (T, 12, 3) keypoints with controlled elbow angles.

    Simulates a rep where elbow goes from elbow_max -> elbow_min -> elbow_max.
    The geometry places shoulder above elbow, and wrist at the desired angle.
    """
    kps = np.zeros((n_frames, 12, 3), dtype=np.float32)

    for t in range(n_frames):
        kps[t, :, 2] = 0.9

        kps[t, 0] = [100.0, 100.0, 0.9]  # left_shoulder
        kps[t, 1] = [200.0, 100.0, 0.9]  # right_shoulder

        # Elbow angle varies over time (down and up)
        progress = t / (n_frames - 1)
        if progress < 0.5:
            angle = elbow_max - (elbow_max - elbow_min) * (progress * 2)
        else:
            angle = elbow_min + (elbow_max - elbow_min) * ((progress - 0.5) * 2)

        # Shoulder at (100, 100), elbow at (100, 150) => vector shoulder->elbow is (0, 50)
        kps[t, 2] = [100.0, 150.0, 0.9]  # left_elbow

        # Place wrist so the angle at elbow (shoulder-elbow-wrist) = desired angle
        # Vector from elbow to shoulder: (0, -50). Rotate by (180 - angle) to get wrist direction.
        half_angle = np.radians(180.0 - angle)
        kps[t, 4] = [
            100.0 + 50.0 * np.sin(half_angle),
            150.0 + 50.0 * np.cos(half_angle),
            0.9,
        ]  # left_wrist

        # Back alignment: shoulder-hip-ankle in a straight line
        kps[t, 6] = [100.0, 250.0, 0.9]  # left_hip
        kps[t, 8] = [100.0, 350.0, 0.9]  # left_knee
        kps[t, 10] = [100.0, 450.0, 0.9] # left_ankle

    return kps


def test_scorer_returns_rep_score():
    scorer = QualityScorer()
    kps = _make_rep_kps(80, 170, 180)
    score = scorer.score_rep(kps)
    assert isinstance(score, RepScore)
    assert 0 <= score.composite <= 100


def test_good_rep_high_score():
    scorer = QualityScorer()
    kps = _make_rep_kps(80, 175, 180)
    score = scorer.score_rep(kps)
    # Good depth + extension should score well
    assert score.depth > 50
    assert score.extension > 50


def test_feedback_returns_list():
    score = RepScore(back_alignment=90, depth=90, extension=90, composite=90)
    feedback = generate_feedback(score)
    assert isinstance(feedback, list)
    assert len(feedback) > 0


def test_poor_form_gets_more_feedback():
    good_score = RepScore(back_alignment=90, depth=90, extension=90, composite=90)
    poor_score = RepScore(back_alignment=30, depth=30, extension=30, composite=30)
    good_fb = generate_feedback(good_score)
    poor_fb = generate_feedback(poor_score)
    assert len(poor_fb) >= len(good_fb)
