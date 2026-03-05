"""Tests for the R3D-18 study module."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from model import PushUpR3D
from datasets import (
    _bbox_from_keypoints,
    _preprocess_full_frame,
    _preprocess_cropped_frame,
)
from data_loader import load_annotations, attach_keypoints


# ---------- Model tests ----------

class TestPushUpR3D:
    def test_frozen_param_count(self):
        model = PushUpR3D(freeze_backbone=True)
        # Only FC layer: 512 * 2 + 2 = 1,026
        assert model.trainable_param_count() == 1026

    def test_unfrozen_has_more_params(self):
        frozen = PushUpR3D(freeze_backbone=True)
        unfrozen_1 = PushUpR3D(freeze_backbone=True)
        unfrozen_1.unfreeze_last_n_blocks(1)
        unfrozen_2 = PushUpR3D(freeze_backbone=True)
        unfrozen_2.unfreeze_last_n_blocks(2)

        p_frozen = frozen.trainable_param_count()
        p_1 = unfrozen_1.trainable_param_count()
        p_2 = unfrozen_2.trainable_param_count()

        assert p_frozen < p_1 < p_2

    def test_forward_shape(self):
        model = PushUpR3D(freeze_backbone=True)
        model.eval()
        x = torch.randn(2, 3, 16, 112, 112)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)

    def test_unfreeze_layers_are_trainable(self):
        model = PushUpR3D(freeze_backbone=True)
        model.unfreeze_last_n_blocks(1)

        for name, param in model.backbone.named_parameters():
            if name.startswith("layer4"):
                assert param.requires_grad, f"{name} should be trainable"
            elif name.startswith("fc"):
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"


# ---------- Dataset helper tests ----------

class TestDatasetHelpers:
    def test_bbox_from_keypoints_basic(self):
        kps = np.zeros((12, 3), dtype=np.float32)
        kps[0] = [100, 50, 0.9]   # left shoulder
        kps[1] = [200, 50, 0.9]   # right shoulder
        kps[6] = [100, 200, 0.9]  # left hip
        kps[7] = [200, 200, 0.9]  # right hip

        x1, y1, x2, y2 = _bbox_from_keypoints(kps, 480, 640)

        assert x1 < 100
        assert y1 < 50
        assert x2 > 200
        assert y2 > 200

    def test_bbox_no_visible_keypoints(self):
        kps = np.zeros((12, 3), dtype=np.float32)
        x1, y1, x2, y2 = _bbox_from_keypoints(kps, 480, 640)
        assert (x1, y1, x2, y2) == (0, 0, 640, 480)

    def test_preprocess_full_frame_shape(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = _preprocess_full_frame(frame)
        assert result.shape == (112, 112, 3)

    def test_preprocess_full_frame_none(self):
        result = _preprocess_full_frame(None)
        assert result.shape == (112, 112, 3)
        assert np.all(result == 0)

    def test_preprocess_cropped_frame_shape(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        kps = np.zeros((12, 3), dtype=np.float32)
        kps[0] = [100, 50, 0.9]
        kps[6] = [100, 200, 0.9]

        result = _preprocess_cropped_frame(frame, kps)
        assert result.shape == (112, 112, 3)


# ---------- Data loader tests ----------

class TestDataLoader:
    def _make_csv(self, tmpdir: Path, rows: list[dict]) -> Path:
        """Write a test annotations CSV."""
        csv_path = tmpdir / "annotations.csv"
        fieldnames = [
            "video_filename", "rep_number", "start_frame", "end_frame",
            "label", "annotator",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return csv_path

    def _make_dummy_video(self, video_dir: Path, filename: str) -> None:
        """Create a minimal dummy video file (empty, just needs to exist)."""
        (video_dir / filename).touch()

    def test_load_annotations_basic(self, tmp_path):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        self._make_dummy_video(video_dir, "test_01.mp4")

        csv_path = self._make_csv(tmp_path, [
            {"video_filename": "test_01.mp4", "rep_number": 1,
             "start_frame": 10, "end_frame": 50, "label": "good", "annotator": "alice"},
            {"video_filename": "test_01.mp4", "rep_number": 2,
             "start_frame": 55, "end_frame": 90, "label": "bad", "annotator": "alice"},
        ])

        reps = load_annotations(csv_path, video_dir)

        assert len(reps) == 2
        assert reps[0]["label"] == 0  # good -> 0
        assert reps[1]["label"] == 1  # bad -> 1
        assert reps[0]["start_frame"] == 10
        assert reps[0]["end_frame"] == 50
        assert reps[0]["video_id"] == "test_01"

    def test_load_annotations_skips_missing_video(self, tmp_path):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        # Don't create the video file

        csv_path = self._make_csv(tmp_path, [
            {"video_filename": "missing.mp4", "rep_number": 1,
             "start_frame": 0, "end_frame": 30, "label": "good", "annotator": "bob"},
        ])

        reps = load_annotations(csv_path, video_dir)
        assert len(reps) == 0

    def test_load_annotations_skips_invalid_range(self, tmp_path):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        self._make_dummy_video(video_dir, "test_01.mp4")

        csv_path = self._make_csv(tmp_path, [
            {"video_filename": "test_01.mp4", "rep_number": 1,
             "start_frame": 50, "end_frame": 10, "label": "good", "annotator": "bob"},
        ])

        reps = load_annotations(csv_path, video_dir)
        assert len(reps) == 0

    def test_attach_keypoints(self, tmp_path):
        kp_dir = tmp_path / "keypoints"
        kp_dir.mkdir()

        # Save dummy keypoints
        kps = np.random.rand(100, 12, 3).astype(np.float32)
        np.save(kp_dir / "test_01.npy", kps)

        reps = [
            {"video_id": "test_01", "start_frame": 10, "end_frame": 30, "label": 0},
        ]
        attach_keypoints(reps, kp_dir)

        assert "keypoints" in reps[0]
        assert reps[0]["keypoints"].shape == (21, 12, 3)  # frames 10..30 inclusive

    def test_attach_keypoints_missing_file(self, tmp_path):
        kp_dir = tmp_path / "keypoints"
        kp_dir.mkdir()

        reps = [
            {"video_id": "nonexistent", "start_frame": 0, "end_frame": 10, "label": 0},
        ]
        attach_keypoints(reps, kp_dir)

        assert "keypoints" not in reps[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
