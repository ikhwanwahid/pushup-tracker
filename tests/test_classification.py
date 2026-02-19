"""Tests for the form classification module."""

import numpy as np
import torch
import pytest

from src.classification.stgcn import build_adjacency_matrix, PushUpSTGCN
from src.classification.video_classifier import PushUpVideoClassifier
from src.classification.datasets import PushUpSkeletonDataset, PushUpVideoDataset


class TestAdjacencyMatrix:
    def test_shape(self):
        A = build_adjacency_matrix()
        assert A.shape == (12, 12)

    def test_symmetric(self):
        A = build_adjacency_matrix()
        torch.testing.assert_close(A, A.T)

    def test_self_loops(self):
        """Diagonal should be nonzero (self-loops present)."""
        A = build_adjacency_matrix()
        assert torch.all(A.diag() > 0)


class TestSTGCN:
    def test_forward_shape(self):
        model = PushUpSTGCN(in_channels=2, num_classes=2)
        x = torch.randn(2, 2, 150, 12)
        out = model(x)
        assert out.shape == (2, 2)

    def test_single_sample(self):
        model = PushUpSTGCN(in_channels=2, num_classes=2)
        x = torch.randn(1, 2, 50, 12)
        out = model(x)
        assert out.shape == (1, 2)

    def test_param_count(self):
        model = PushUpSTGCN(in_channels=2, num_classes=2, dropout=0.2)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Should be roughly ~245K, check it's in a reasonable range
        assert 100_000 < n_params < 500_000, f"Got {n_params} params"


class TestR3D:
    def test_forward_shape(self):
        model = PushUpVideoClassifier(freeze_backbone=True, num_classes=2)
        x = torch.randn(2, 3, 16, 112, 112)
        out = model(x)
        assert out.shape == (2, 2)

    def test_frozen_params(self):
        model = PushUpVideoClassifier(freeze_backbone=True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Only FC layer: 512*2 + 2 = 1026
        assert trainable == 1026, f"Got {trainable} trainable params"


class TestSkeletonDataset:
    def test_shape(self, tmp_path):
        """Verify output shape from skeleton dataset."""
        # Create fake keypoints and manifest
        kps = np.random.rand(80, 12, 3).astype(np.float32)
        # Set reasonable shoulder/hip values so torso_normalize doesn't divide by zero
        kps[:, 0, :2] = [100, 50]   # left_shoulder
        kps[:, 1, :2] = [200, 50]   # right_shoulder
        kps[:, 6, :2] = [100, 200]  # left_hip
        kps[:, 7, :2] = [200, 200]  # right_hip
        kps[:, :, 2] = 0.9  # confidence

        np.save(tmp_path / "test_video.npy", kps)

        manifest = {
            "test_video": {"label": "correct", "original_path": "test.mp4"}
        }

        dataset = PushUpSkeletonDataset(
            manifest=manifest,
            keypoint_dir=tmp_path,
            video_ids=["test_video"],
            max_frames=150,
        )

        tensor, label = dataset[0]
        assert tensor.shape == (2, 150, 12)
        assert label == 0  # correct

    def test_incorrect_label(self, tmp_path):
        kps = np.random.rand(50, 12, 3).astype(np.float32)
        kps[:, 0, :2] = [100, 50]
        kps[:, 1, :2] = [200, 50]
        kps[:, 6, :2] = [100, 200]
        kps[:, 7, :2] = [200, 200]
        np.save(tmp_path / "wrong_video.npy", kps)

        manifest = {
            "wrong_video": {"label": "incorrect", "original_path": "wrong.mp4"}
        }

        dataset = PushUpSkeletonDataset(
            manifest=manifest,
            keypoint_dir=tmp_path,
            video_ids=["wrong_video"],
            max_frames=150,
        )

        _, label = dataset[0]
        assert label == 1  # incorrect


class TestVideoDataset:
    def test_shape(self, tmp_path):
        """Verify output shape from video dataset using a synthetic video."""
        # Create a tiny synthetic video
        video_path = tmp_path / "Correct sequence" / "test.mp4"
        video_path.parent.mkdir(parents=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 25.0, (640, 360))
        for _ in range(30):
            frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        manifest = {
            "test_vid": {
                "label": "correct",
                "original_path": "Correct sequence/test.mp4",
            }
        }

        dataset = PushUpVideoDataset(
            manifest=manifest,
            video_dir=tmp_path,
            video_ids=["test_vid"],
            n_frames=16,
        )

        tensor, label = dataset[0]
        assert tensor.shape == (3, 16, 112, 112)
        assert label == 0


# Import cv2 at module level for the video test
import cv2
