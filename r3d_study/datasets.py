"""Dataset classes for R3D-18 study.

Two variants:
  - PushUpRepVideoDataset: full-frame (resize + crop)
  - PushUpRepCroppedVideoDataset: YOLO-crop (keypoint bbox + resize)

Both support train-time augmentation (horizontal flip, random crop,
temporal jitter, color jitter).
"""

import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Kinetics-400 normalization stats
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

MIN_KP_CONFIDENCE = 0.3
BBOX_PAD_RATIO = 0.25  # 25% padding around keypoint bounding box

# Global frame cache: (video_path, frame_idx) -> BGR numpy array
_frame_cache: dict[tuple[str, int], np.ndarray] = {}


def preload_videos(rep_segments: list[dict], n_frames: int = 16, jitter: int = 2) -> None:
    """Pre-decode only the frames needed for each rep (memory-efficient).

    Samples n_frames per rep using uniform spacing, plus ±jitter frames
    to cover augmentation jitter during training.
    """
    needed: dict[str, set[int]] = {}
    for rep in rep_segments:
        path = rep["video_path"]
        start, end = rep["start_frame"], rep["end_frame"]
        indices = _uniform_sample_indices(start, end, n_frames)
        if path not in needed:
            needed[path] = set()
        for i in indices:
            for offset in range(-jitter, jitter + 1):
                idx = max(start, min(end, int(i) + offset))
                needed[path].add(idx)

    total_frames = sum(len(idxs) for idxs in needed.values())
    logger.info("Pre-loading %d frames from %d videos...", total_frames, len(needed))

    for path in sorted(needed):
        frame_indices = sorted(needed[path])
        to_load = [i for i in frame_indices if (path, i) not in _frame_cache]
        if not to_load:
            continue

        cap = cv2.VideoCapture(path)
        for idx in to_load:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                _frame_cache[(path, idx)] = frame
        cap.release()

    logger.info("Frame cache: %d frames loaded", len(_frame_cache))


def _uniform_sample_indices(start: int, end: int, n: int) -> np.ndarray:
    """Generate n uniformly spaced frame indices in [start, end]."""
    return np.linspace(start, end, n).astype(int)


def _jittered_sample_indices(
    start: int, end: int, n: int, jitter: int = 2,
) -> np.ndarray:
    """Uniform sample with random per-frame jitter (clamped to [start, end])."""
    base = np.linspace(start, end, n).astype(int)
    offsets = np.random.randint(-jitter, jitter + 1, size=n)
    return np.clip(base + offsets, start, end)


def _read_frames(video_path: str, frame_indices: np.ndarray) -> list[np.ndarray]:
    """Read specific frames — from cache if available, else from disk."""
    frames = []
    cap = None

    for idx in frame_indices:
        cached = _frame_cache.get((video_path, int(idx)))
        if cached is not None:
            frames.append(cached)
        else:
            if cap is None:
                cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            frames.append(frame if ret else None)

    if cap is not None:
        cap.release()
    return frames


def _preprocess_full_frame(
    frame: np.ndarray | None, random_crop: bool = False,
) -> np.ndarray:
    """Resize to 128x171, crop to 112x112, BGR->RGB."""
    if frame is None:
        return np.zeros((112, 112, 3), dtype=np.uint8)

    frame = cv2.resize(frame, (171, 128))

    if random_crop:
        y_max = 128 - 112
        x_max = 171 - 112
        y_off = random.randint(0, y_max)
        x_off = random.randint(0, x_max)
    else:
        y_off = (128 - 112) // 2
        x_off = (171 - 112) // 2

    frame = frame[y_off:y_off + 112, x_off:x_off + 112]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _bbox_from_keypoints(
    keypoints: np.ndarray, frame_h: int, frame_w: int,
) -> tuple[int, int, int, int]:
    """Compute bounding box from keypoints with padding.

    Args:
        keypoints: (12, 3) unified keypoints [x, y, confidence].
        frame_h: Frame height for clamping.
        frame_w: Frame width for clamping.

    Returns:
        (x1, y1, x2, y2) bounding box in pixel coordinates.
    """
    visible = keypoints[keypoints[:, 2] > MIN_KP_CONFIDENCE]

    if len(visible) < 2:
        return 0, 0, frame_w, frame_h

    x_min, y_min = visible[:, 0].min(), visible[:, 1].min()
    x_max, y_max = visible[:, 0].max(), visible[:, 1].max()

    w, h = x_max - x_min, y_max - y_min
    pad_x = w * BBOX_PAD_RATIO
    pad_y = h * BBOX_PAD_RATIO

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(frame_w, int(x_max + pad_x))
    y2 = min(frame_h, int(y_max + pad_y))

    if x2 - x1 < 10 or y2 - y1 < 10:
        return 0, 0, frame_w, frame_h

    return x1, y1, x2, y2


def _preprocess_cropped_frame(
    frame: np.ndarray | None,
    keypoints: np.ndarray,
) -> np.ndarray:
    """Crop frame to person bbox (from keypoints), resize to 112x112, BGR->RGB."""
    if frame is None:
        return np.zeros((112, 112, 3), dtype=np.uint8)

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _bbox_from_keypoints(keypoints, h, w)

    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, (112, 112))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _apply_augmentation(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Apply consistent augmentations across all frames in a clip.

    Applied randomly per clip (not per frame) so temporal consistency is kept.
    """
    # Horizontal flip (50% chance)
    if random.random() < 0.5:
        frames = [np.fliplr(f).copy() for f in frames]

    # Color jitter — small brightness/contrast shift
    if random.random() < 0.5:
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        frames = [
            np.clip(contrast * f.astype(np.float32) + (brightness - 1.0) * 128, 0, 255).astype(np.uint8)
            for f in frames
        ]

    return frames


def _to_tensor(frames: list[np.ndarray]) -> np.ndarray:
    """Stack frames and normalize to Kinetics-400 stats.

    Returns:
        (3, T, 112, 112) float32 array.
    """
    video = np.stack(frames).astype(np.float32) / 255.0  # (T, 112, 112, 3)
    video = video.transpose(3, 0, 1, 2)  # (3, T, 112, 112)

    mean = KINETICS_MEAN.reshape(3, 1, 1, 1)
    std = KINETICS_STD.reshape(3, 1, 1, 1)
    return (video - mean) / std


class PushUpRepVideoDataset(Dataset):
    """Per-rep video dataset — full-frame preprocessing.

    Each rep dict must have: video_path, start_frame, end_frame, label.

    Args:
        rep_segments: List of rep dicts.
        n_frames: Number of frames to sample per rep.
        augment: If True, apply random augmentations (for training).

    Returns:
        (3, n_frames, 112, 112) float32 tensor, label (0=good, 1=bad)
    """

    def __init__(
        self, rep_segments: list[dict], n_frames: int = 16, augment: bool = False,
    ):
        self.rep_segments = rep_segments
        self.n_frames = n_frames
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rep_segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rep = self.rep_segments[idx]

        if self.augment:
            indices = _jittered_sample_indices(
                rep["start_frame"], rep["end_frame"], self.n_frames,
            )
        else:
            indices = _uniform_sample_indices(
                rep["start_frame"], rep["end_frame"], self.n_frames,
            )

        raw_frames = _read_frames(rep["video_path"], indices)
        processed = [
            _preprocess_full_frame(f, random_crop=self.augment)
            for f in raw_frames
        ]

        if self.augment:
            processed = _apply_augmentation(processed)

        tensor = _to_tensor(processed)
        return torch.from_numpy(tensor), rep["label"]


class PushUpRepCroppedVideoDataset(Dataset):
    """Per-rep video dataset — YOLO-crop preprocessing.

    Uses saved keypoints to compute per-frame bounding boxes, crops frames
    to the detected person, resizes to 112x112.

    Each rep dict must have: video_path, start_frame, end_frame, label, keypoints.

    Args:
        rep_segments: List of rep dicts.
        n_frames: Number of frames to sample per rep.
        augment: If True, apply random augmentations (for training).

    Returns:
        (3, n_frames, 112, 112) float32 tensor, label (0=good, 1=bad)
    """

    def __init__(
        self, rep_segments: list[dict], n_frames: int = 16, augment: bool = False,
    ):
        self.rep_segments = rep_segments
        self.n_frames = n_frames
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rep_segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rep = self.rep_segments[idx]
        start, end = rep["start_frame"], rep["end_frame"]
        kps = rep["keypoints"]  # (T_rep, 12, 3)

        if self.augment:
            indices = _jittered_sample_indices(start, end, self.n_frames)
        else:
            indices = _uniform_sample_indices(start, end, self.n_frames)

        raw_frames = _read_frames(rep["video_path"], indices)

        processed = []
        for i, (frame, fi) in enumerate(zip(raw_frames, indices)):
            kp_idx = min(fi - start, len(kps) - 1)
            kp_idx = max(0, kp_idx)
            processed.append(_preprocess_cropped_frame(frame, kps[kp_idx]))

        if self.augment:
            processed = _apply_augmentation(processed)

        tensor = _to_tensor(processed)
        return torch.from_numpy(tensor), rep["label"]
