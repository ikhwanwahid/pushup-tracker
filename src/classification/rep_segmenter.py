"""Rep segmentation utility for per-rep form classification.

Segments videos into individual reps using the state machine,
then provides per-rep keypoints, video boundaries, and angle features.
"""

import logging
from pathlib import Path

import numpy as np

from src.counting.state_machine import PushUpStateMachine
from src.features.angles import compute_angle_sequence

logger = logging.getLogger(__name__)

MIN_REP_FRAMES = 10


def segment_all_videos(
    manifest: dict,
    keypoint_dir: str | Path,
    min_rep_frames: int = MIN_REP_FRAMES,
    down_threshold: float = 90.0,
    up_threshold: float = 160.0,
) -> list[dict]:
    """Segment all videos into individual reps.

    For each video, runs the state machine to find rep boundaries,
    then extracts per-rep keypoint slices with inherited labels.

    Args:
        manifest: Video manifest dict (video_id -> metadata with "label").
        keypoint_dir: Path to directory containing {video_id}.npy files.
        min_rep_frames: Minimum frames for a valid rep (filters degenerate reps).
        down_threshold: Elbow angle for "down" position (degrees).
        up_threshold: Elbow angle for "up" position (degrees).

    Returns:
        List of rep dicts, each containing:
            - video_id: Source video ID
            - rep_idx: Rep index within the video (0-based)
            - start_frame: Start frame index in the original video
            - end_frame: End frame index in the original video
            - label: 0 (correct) or 1 (incorrect), inherited from video
            - keypoints: (T_rep, 12, 3) array of keypoints for this rep
    """
    keypoint_dir = Path(keypoint_dir)
    all_reps = []

    for video_id in sorted(manifest.keys()):
        npy_path = keypoint_dir / f"{video_id}.npy"
        if not npy_path.exists():
            continue

        kps = np.load(npy_path)  # (T, 12, 3)
        label = 0 if manifest[video_id]["label"] == "correct" else 1

        sm = PushUpStateMachine(
            down_threshold=down_threshold, up_threshold=up_threshold
        )
        boundaries = sm.segment_sequence(kps)

        for rep_idx, (start, end) in enumerate(boundaries):
            n_frames = end - start + 1
            if n_frames < min_rep_frames:
                continue

            all_reps.append({
                "video_id": video_id,
                "rep_idx": rep_idx,
                "start_frame": start,
                "end_frame": end,
                "label": label,
                "keypoints": kps[start:end + 1],
            })

    logger.info(
        "Segmented %d reps from %d videos (min_frames=%d)",
        len(all_reps), len(manifest), min_rep_frames,
    )
    return all_reps


def compute_rep_features(rep_kps: np.ndarray) -> np.ndarray:
    """Compute per-rep angle statistics for the baseline classifier.

    Args:
        rep_kps: (T, 12, 3) keypoints for a single rep.

    Returns:
        (16,) array: mean/min/max/range for elbow, back, hip, knee angles.
    """
    angles = compute_angle_sequence(rep_kps)  # (T, 4)

    features = []
    for col in range(4):
        col_vals = angles[:, col]
        features.extend([
            col_vals.mean(),
            col_vals.min(),
            col_vals.max(),
            col_vals.max() - col_vals.min(),  # range
        ])

    return np.array(features, dtype=np.float32)
