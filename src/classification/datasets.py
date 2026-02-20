"""Dataset classes for push-up form classification.

Per-video classes (used by FormChecker rolling buffer):
  PushUpVideoDataset, PushUpSkeletonDataset

Per-rep classes (used by training pipeline):
  PushUpRepSkeletonDataset, PushUpRepVideoDataset
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.normalize import torso_normalize

# Kinetics-400 normalization stats
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]


class PushUpVideoDataset(Dataset):
    """Load raw video frames for the 3D CNN classifier.

    Uniform-samples n_frames frames, resizes to 128x171, center-crops to 112x112,
    and normalizes with Kinetics-400 statistics.

    Returns:
        (3, n_frames, 112, 112) float32 tensor, label (0=correct, 1=incorrect)
    """

    def __init__(
        self,
        manifest: dict,
        video_dir: str | Path,
        video_ids: list[str],
        n_frames: int = 16,
    ):
        self.manifest = manifest
        self.video_dir = Path(video_dir)
        self.video_ids = video_ids
        self.n_frames = n_frames

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        vid_id = self.video_ids[idx]
        meta = self.manifest[vid_id]

        label = 0 if meta["label"] == "correct" else 1

        # Resolve video path
        video_path = self.video_dir / meta["original_path"]

        # Load frames
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Uniform sample frame indices
        indices = np.linspace(0, total_frames - 1, self.n_frames).astype(int)

        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # Fallback: black frame
                frame = np.zeros((112, 112, 3), dtype=np.uint8)
            else:
                # Resize to 128x171 (short side 128)
                frame = cv2.resize(frame, (171, 128))
                # Center crop to 112x112
                y_off = (128 - 112) // 2
                x_off = (171 - 112) // 2
                frame = frame[y_off:y_off + 112, x_off:x_off + 112]
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        # Stack: (T, H, W, C) -> (C, T, H, W)
        video = np.stack(frames).astype(np.float32) / 255.0  # (T, 112, 112, 3)
        video = video.transpose(3, 0, 1, 2)  # (3, T, 112, 112)

        # Normalize with Kinetics stats
        mean = np.array(KINETICS_MEAN, dtype=np.float32).reshape(3, 1, 1, 1)
        std = np.array(KINETICS_STD, dtype=np.float32).reshape(3, 1, 1, 1)
        video = (video - mean) / std

        return torch.from_numpy(video), label


class PushUpSkeletonDataset(Dataset):
    """Load keypoint sequences for ST-GCN classification.

    Applies torso normalization, centers by hip midpoint,
    extracts (x,y) only, and pads/truncates to max_frames.

    Returns:
        (2, max_frames, 12) float32 tensor, label (0=correct, 1=incorrect)
    """

    def __init__(
        self,
        manifest: dict,
        keypoint_dir: str | Path,
        video_ids: list[str],
        max_frames: int = 150,
        normalize: bool = True,
    ):
        self.manifest = manifest
        self.keypoint_dir = Path(keypoint_dir)
        self.video_ids = video_ids
        self.max_frames = max_frames
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        vid_id = self.video_ids[idx]
        meta = self.manifest[vid_id]

        label = 0 if meta["label"] == "correct" else 1

        # Load keypoints: (T, 12, 3)
        kps = np.load(self.keypoint_dir / f"{vid_id}.npy")
        T = len(kps)

        # Process each frame
        processed = []
        for t in range(T):
            frame_kps = kps[t].copy()  # (12, 3)

            if self.normalize:
                frame_kps = torso_normalize(frame_kps)

            # Center by hip midpoint for translation invariance
            hip_mid = (frame_kps[6, :2] + frame_kps[7, :2]) / 2.0
            frame_kps[:, :2] -= hip_mid

            # Extract (x, y) only
            processed.append(frame_kps[:, :2])  # (12, 2)

        processed = np.stack(processed)  # (T, 12, 2)

        # Pad or truncate to max_frames
        if T < self.max_frames:
            pad = np.zeros((self.max_frames - T, 12, 2), dtype=np.float32)
            processed = np.concatenate([processed, pad], axis=0)
        else:
            processed = processed[:self.max_frames]

        # Reshape to (2, max_frames, 12) for ST-GCN input
        processed = processed.astype(np.float32)
        tensor = torch.from_numpy(processed).permute(2, 0, 1)  # (2, T, 12)

        return tensor, label


def _process_skeleton_frames(kps: np.ndarray, max_frames: int, normalize: bool = True) -> np.ndarray:
    """Shared preprocessing for skeleton data: normalize, center, extract xy, pad/truncate.

    Args:
        kps: (T, 12, 3) keypoints.
        max_frames: Target sequence length.
        normalize: Whether to apply torso normalization.

    Returns:
        (2, max_frames, 12) float32 array.
    """
    T = len(kps)
    processed = []
    for t in range(T):
        frame_kps = kps[t].copy()
        if normalize:
            frame_kps = torso_normalize(frame_kps)
        hip_mid = (frame_kps[6, :2] + frame_kps[7, :2]) / 2.0
        frame_kps[:, :2] -= hip_mid
        processed.append(frame_kps[:, :2])

    processed = np.stack(processed)  # (T, 12, 2)

    if T < max_frames:
        pad = np.zeros((max_frames - T, 12, 2), dtype=np.float32)
        processed = np.concatenate([processed, pad], axis=0)
    else:
        processed = processed[:max_frames]

    return processed.astype(np.float32).transpose(2, 0, 1)  # (2, max_frames, 12)


def _load_video_frames(
    video_path: str, start_frame: int, end_frame: int, n_frames: int,
) -> np.ndarray:
    """Load and preprocess video frames from a frame range.

    Returns:
        (3, n_frames, 112, 112) float32 array, Kinetics-normalized.
    """
    cap = cv2.VideoCapture(video_path)
    indices = np.linspace(start_frame, end_frame, n_frames).astype(int)

    frames = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((112, 112, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (171, 128))
            y_off = (128 - 112) // 2
            x_off = (171 - 112) // 2
            frame = frame[y_off:y_off + 112, x_off:x_off + 112]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    video = np.stack(frames).astype(np.float32) / 255.0
    video = video.transpose(3, 0, 1, 2)  # (3, T, 112, 112)

    mean = np.array(KINETICS_MEAN, dtype=np.float32).reshape(3, 1, 1, 1)
    std = np.array(KINETICS_STD, dtype=np.float32).reshape(3, 1, 1, 1)
    return (video - mean) / std


class PushUpRepSkeletonDataset(Dataset):
    """Per-rep skeleton dataset for ST-GCN.

    Takes pre-segmented rep dicts from rep_segmenter.segment_all_videos().

    Returns:
        (2, max_frames, 12) float32 tensor, label (0=correct, 1=incorrect)
    """

    def __init__(self, rep_segments: list[dict], max_frames: int = 64, normalize: bool = True):
        self.rep_segments = rep_segments
        self.max_frames = max_frames
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.rep_segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rep = self.rep_segments[idx]
        kps = rep["keypoints"]  # (T_rep, 12, 3)
        label = rep["label"]

        tensor = _process_skeleton_frames(kps, self.max_frames, self.normalize)
        return torch.from_numpy(tensor), label


class PushUpRepVideoDataset(Dataset):
    """Per-rep video dataset for R3D-18.

    Takes pre-segmented rep dicts with video path and frame boundaries.

    Returns:
        (3, n_frames, 112, 112) float32 tensor, label (0=correct, 1=incorrect)
    """

    def __init__(self, rep_segments: list[dict], n_frames: int = 16):
        self.rep_segments = rep_segments
        self.n_frames = n_frames

    def __len__(self) -> int:
        return len(self.rep_segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rep = self.rep_segments[idx]
        label = rep["label"]

        video = _load_video_frames(
            rep["video_path"], rep["start_frame"], rep["end_frame"], self.n_frames,
        )
        return torch.from_numpy(video), label
