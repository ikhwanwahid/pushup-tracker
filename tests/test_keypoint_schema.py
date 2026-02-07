"""Tests for the unified keypoint schema and mapping."""

import numpy as np

from src.pose_estimation.keypoint_schema import (
    NUM_UNIFIED_JOINTS,
    UNIFIED_JOINTS,
    YOLO_TO_UNIFIED,
    MEDIAPIPE_TO_UNIFIED,
    MOVENET_TO_UNIFIED,
    map_to_unified,
)


def test_unified_joint_count():
    assert NUM_UNIFIED_JOINTS == 12
    assert len(UNIFIED_JOINTS) == 12


def test_yolo_mapping_covers_all_unified():
    """YOLO mapping should map to all 12 unified joints."""
    assert set(YOLO_TO_UNIFIED.values()) == set(range(12))


def test_mediapipe_mapping_covers_all_unified():
    assert set(MEDIAPIPE_TO_UNIFIED.values()) == set(range(12))


def test_movenet_mapping_covers_all_unified():
    assert set(MOVENET_TO_UNIFIED.values()) == set(range(12))


def test_map_to_unified_shape():
    native_kps = np.random.rand(17, 3).astype(np.float32)
    unified = map_to_unified(native_kps, YOLO_TO_UNIFIED)
    assert unified.shape == (12, 3)


def test_map_to_unified_values():
    """Check that specific native indices map to the right unified slots."""
    native_kps = np.zeros((17, 3), dtype=np.float32)
    # Set left shoulder (YOLO index 5) to known value
    native_kps[5] = [100.0, 200.0, 0.9]
    unified = map_to_unified(native_kps, YOLO_TO_UNIFIED)
    # Left shoulder should be at unified index 0
    np.testing.assert_array_almost_equal(unified[0], [100.0, 200.0, 0.9])


def test_map_to_unified_zeros_for_unmapped():
    """Unmapped joints should remain zero."""
    native_kps = np.ones((17, 3), dtype=np.float32)
    unified = map_to_unified(native_kps, YOLO_TO_UNIFIED)
    # All mapped joints should be 1.0; none should be zero since all 12 are mapped
    assert np.all(unified != 0)


def test_map_to_unified_mediapipe():
    native_kps = np.zeros((33, 3), dtype=np.float32)
    native_kps[11] = [50.0, 60.0, 0.8]  # left_shoulder -> unified 0
    unified = map_to_unified(native_kps, MEDIAPIPE_TO_UNIFIED)
    np.testing.assert_array_almost_equal(unified[0], [50.0, 60.0, 0.8])
