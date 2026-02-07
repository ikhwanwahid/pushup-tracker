"""Tests for joint angle calculations."""

import math

import numpy as np
import pytest

from src.features.angles import (
    calculate_angle,
    compute_elbow_angle,
    compute_back_alignment,
    compute_all_angles,
    compute_angle_sequence,
)


def test_calculate_angle_straight():
    """180 degrees: three points in a line."""
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 0.0])
    p3 = np.array([2.0, 0.0])
    assert abs(calculate_angle(p1, p2, p3) - 180.0) < 1e-5


def test_calculate_angle_right():
    """90 degrees: L-shape."""
    p1 = np.array([0.0, 1.0])
    p2 = np.array([0.0, 0.0])
    p3 = np.array([1.0, 0.0])
    assert abs(calculate_angle(p1, p2, p3) - 90.0) < 1e-5


def test_calculate_angle_zero_length():
    """Degenerate case: coincident points."""
    p1 = np.array([1.0, 1.0])
    p2 = np.array([1.0, 1.0])
    p3 = np.array([2.0, 2.0])
    assert calculate_angle(p1, p2, p3) == 0.0


def test_compute_elbow_angle():
    """Elbow angle from unified keypoints."""
    kps = np.zeros((12, 3), dtype=np.float32)
    # Left side: shoulder=0, elbow=2, wrist=4
    kps[0] = [0.0, 100.0, 0.9]  # left_shoulder
    kps[2] = [0.0, 50.0, 0.9]   # left_elbow
    kps[4] = [50.0, 50.0, 0.9]  # left_wrist
    angle = compute_elbow_angle(kps, side="left")
    assert abs(angle - 90.0) < 1e-3


def test_compute_all_angles_keys():
    kps = np.random.rand(12, 3).astype(np.float32)
    angles = compute_all_angles(kps)
    assert "elbow_angle" in angles
    assert "back_alignment" in angles
    assert "hip_angle" in angles
    assert "knee_angle" in angles


def test_compute_angle_sequence_shape():
    T = 10
    kps_seq = np.random.rand(T, 12, 3).astype(np.float32)
    angles = compute_angle_sequence(kps_seq)
    assert angles.shape == (T, 4)
