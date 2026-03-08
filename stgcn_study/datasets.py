"""Dataset classes for ST-GCN push-up form classification.

Provides skeleton-based datasets that load pre-extracted YOLO keypoints,
normalize them (torso-normalize + hip-center), and return tensors
suitable for the ST-GCN model.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# Joint indices (unified 12-joint format)
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

# Pairs of (left, right) joint indices for horizontal flip
LR_SWAP_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]


# ============================================================
# Normalization
# ============================================================


def torso_normalize(unified_kps: np.ndarray) -> np.ndarray:
    """Normalize keypoints by dividing coordinates by torso length.

    Torso length = average of left and right shoulder-hip distances.
    Makes the representation invariant to subject distance from camera.

    Args:
        unified_kps: (12, 3) unified keypoints [x, y, confidence].

    Returns:
        (12, 3) normalized keypoints. Confidence values are preserved.
    """
    kps = unified_kps.copy()

    ls = kps[UNIFIED_JOINT_INDEX["left_shoulder"], :2]
    rs = kps[UNIFIED_JOINT_INDEX["right_shoulder"], :2]
    lh = kps[UNIFIED_JOINT_INDEX["left_hip"], :2]
    rh = kps[UNIFIED_JOINT_INDEX["right_hip"], :2]

    left_torso = np.linalg.norm(ls - lh)
    right_torso = np.linalg.norm(rs - rh)
    torso_length = (left_torso + right_torso) / 2.0

    if torso_length < 1e-6:
        return kps

    kps[:, :2] /= torso_length
    return kps


def hip_center(unified_kps: np.ndarray) -> np.ndarray:
    """Center keypoints by subtracting hip midpoint from all coordinates.

    Args:
        unified_kps: (12, 3) unified keypoints [x, y, confidence].

    Returns:
        (12, 3) centered keypoints.
    """
    kps = unified_kps.copy()
    lh = kps[UNIFIED_JOINT_INDEX["left_hip"], :2]
    rh = kps[UNIFIED_JOINT_INDEX["right_hip"], :2]
    center = (lh + rh) / 2.0
    kps[:, :2] -= center
    return kps


# ============================================================
# Temporal sampling
# ============================================================


def _uniform_sample_indices(total_frames: int, n_frames: int) -> np.ndarray:
    """Uniformly sample n_frames indices from [0, total_frames-1]."""
    if total_frames <= 0:
        return np.zeros(n_frames, dtype=int)
    return np.linspace(0, total_frames - 1, n_frames).astype(int)


def _jittered_sample_indices(
    total_frames: int, n_frames: int, jitter: float = 0.5,
) -> np.ndarray:
    """Sample n_frames with random jitter for augmentation."""
    if total_frames <= 1:
        return np.zeros(n_frames, dtype=int)
    base = np.linspace(0, total_frames - 1, n_frames)
    step = (total_frames - 1) / max(n_frames - 1, 1)
    noise = np.random.uniform(-jitter * step, jitter * step, n_frames)
    indices = np.clip(base + noise, 0, total_frames - 1).astype(int)
    return indices


# ============================================================
# Preprocessing pipeline
# ============================================================


def preprocess_keypoints(
    kps_rep: np.ndarray,
    n_frames: int = 32,
    in_channels: int = 2,
    augment: bool = False,
) -> np.ndarray:
    """Preprocess a rep's keypoint sequence for ST-GCN input.

    Pipeline per frame:
        1. Torso-normalize (scale invariance)
        2. Hip-center (translation invariance)
        3. Extract channels (x,y or x,y,conf)

    Args:
        kps_rep: (T, 12, 3) keypoint sequence for one rep.
        n_frames: Number of frames to sample.
        in_channels: 2 for (x,y), 3 for (x,y,confidence).
        augment: If True, apply augmentations.

    Returns:
        (in_channels, n_frames, 12) float32 array ready for ST-GCN.
    """
    T = len(kps_rep)

    # Temporal sampling
    if augment:
        indices = _jittered_sample_indices(T, n_frames)
    else:
        indices = _uniform_sample_indices(T, n_frames)

    sampled = kps_rep[indices]  # (n_frames, 12, 3)

    # Per-frame normalization
    normalized = np.zeros_like(sampled)
    for t in range(n_frames):
        frame = torso_normalize(sampled[t])
        frame = hip_center(frame)
        normalized[t] = frame

    # Augmentation: horizontal flip (negate x, swap L/R joints)
    if augment and np.random.random() < 0.5:
        normalized[:, :, 0] *= -1  # negate x
        for li, ri in LR_SWAP_PAIRS:
            normalized[:, [li, ri]] = normalized[:, [ri, li]]

    # Augmentation: small Gaussian noise on coordinates
    if augment:
        noise = np.random.normal(0, 0.01, normalized[:, :, :2].shape)
        normalized[:, :, :2] += noise.astype(np.float32)

    # Extract channels: (n_frames, 12, C) -> (C, n_frames, 12)
    output = normalized[:, :, :in_channels]  # (n_frames, 12, in_channels)
    output = output.transpose(2, 0, 1)       # (in_channels, n_frames, 12)

    return output.astype(np.float32)


# ============================================================
# Dataset class
# ============================================================


class PushUpRepSkeletonDataset(Dataset):
    """Dataset for per-rep skeleton classification.

    Each sample is a preprocessed keypoint sequence for one rep.

    Args:
        rep_segments: List of rep dicts, each must have "keypoints" (T, 12, 3)
            and "label" (0=good, 1=bad).
        n_frames: Number of frames to sample per rep.
        in_channels: 2 for (x,y), 3 for (x,y,confidence).
        augment: Whether to apply augmentations.
    """

    def __init__(
        self,
        rep_segments: list[dict],
        n_frames: int = 32,
        in_channels: int = 2,
        augment: bool = False,
    ):
        self.reps = [r for r in rep_segments if "keypoints" in r]
        self.n_frames = n_frames
        self.in_channels = in_channels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.reps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rep = self.reps[idx]
        kps = rep["keypoints"]  # (T, 12, 3)
        label = rep["label"]

        tensor = preprocess_keypoints(
            kps, self.n_frames, self.in_channels, self.augment,
        )
        return torch.from_numpy(tensor), label
