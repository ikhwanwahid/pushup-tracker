"""Data loading from manually annotated CSV.

Reads per-rep annotations with human-labeled frame boundaries,
resolves video paths, and loads pre-extracted YOLO keypoints.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_annotations(
    file_path: str | Path,
    video_dir: str | Path,
) -> list[dict]:
    """Load per-rep annotations from CSV or Excel (.xlsx).

    Expected columns:
        video_filename, rep_number, start_frame, end_frame, label

    Args:
        file_path: Path to annotations CSV or .xlsx file.
        video_dir: Directory containing the video files.

    Returns:
        List of rep dicts, each with:
            - video_id: filename without extension
            - video_path: full path to video file
            - rep_number: 1-based rep index
            - start_frame: annotated start frame
            - end_frame: annotated end frame
            - label: 0 (good) or 1 (bad)
    """
    file_path = Path(file_path)
    video_dir = Path(video_dir)

    if file_path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    reps = []
    skipped = 0

    for _, row in df.iterrows():
        if pd.isna(row.get("video_filename")) or pd.isna(row.get("label")):
            skipped += 1
            continue
        if pd.isna(row.get("start_frame")) or pd.isna(row.get("end_frame")):
            skipped += 1
            continue

        filename = str(row["video_filename"]).strip()
        video_path = video_dir / filename

        if not video_path.exists():
            logger.warning("Video not found, skipping: %s", video_path)
            skipped += 1
            continue

        start = int(row["start_frame"])
        end = int(row["end_frame"])

        if end <= start:
            logger.warning(
                "Invalid frame range (%d-%d) for %s rep %s, skipping",
                start, end, filename, row.get("rep_number", "?"),
            )
            skipped += 1
            continue

        label_str = str(row["label"]).strip().lower()
        label = 0 if label_str == "good" else 1

        video_id = Path(filename).stem

        reps.append({
            "video_id": video_id,
            "video_path": str(video_path),
            "rep_number": int(row.get("rep_number", 1)),
            "start_frame": start,
            "end_frame": end,
            "label": label,
        })

    logger.info(
        "Loaded %d reps from %s (skipped %d)",
        len(reps), file_path.name, skipped,
    )
    return reps


def extract_keypoints(
    rep_segments: list[dict],
    output_dir: str | Path,
    model_path: str = "yolo11s-pose.pt",
) -> None:
    """Extract YOLO keypoints for all videos referenced in rep_segments.

    Only processes videos that don't already have a .npy file.

    Args:
        rep_segments: List of rep dicts from load_annotations().
        output_dir: Directory to save {video_id}.npy files.
        model_path: Path to YOLO pose model weights.
    """
    from ultralytics import YOLO
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = {}
    for rep in rep_segments:
        vid_id = rep["video_id"]
        if vid_id not in videos:
            videos[vid_id] = rep["video_path"]

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = YOLO(model_path)

    yolo_to_unified = {
        5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 10: 5,
        11: 6, 12: 7, 13: 8, 14: 9, 15: 10, 16: 11,
    }

    for vid_id, video_path in sorted(videos.items()):
        npy_path = output_dir / f"{vid_id}.npy"
        if npy_path.exists():
            logger.info("Keypoints already exist: %s", npy_path.name)
            continue

        logger.info("Extracting keypoints: %s", vid_id)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_kps = []

        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                all_kps.append(np.zeros((12, 3), dtype=np.float32))
                continue

            results = model(frame, device=device, conf=0.5, verbose=False)
            result = results[0]

            unified = np.zeros((12, 3), dtype=np.float32)
            if result.keypoints is not None and len(result.keypoints):
                kps_data = result.keypoints.data.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.zeros(len(kps_data))
                best_idx = int(np.argmax(scores))
                native_kps = kps_data[best_idx]

                for native_idx, unified_idx in yolo_to_unified.items():
                    if native_idx < len(native_kps):
                        unified[unified_idx] = native_kps[native_idx]

            all_kps.append(unified)

        cap.release()

        kps_array = np.stack(all_kps)
        np.save(npy_path, kps_array)
        logger.info("  Saved: %s (%d frames)", npy_path.name, len(all_kps))

    logger.info("Keypoint extraction complete: %d videos", len(videos))


def attach_keypoints(
    rep_segments: list[dict],
    keypoint_dir: str | Path,
) -> None:
    """Load pre-extracted keypoints and attach to rep dicts (in-place).

    Adds a "keypoints" key to each rep dict containing the (T_rep, 12, 3)
    keypoint array for that rep's frame range.

    Args:
        rep_segments: List of rep dicts (modified in-place).
        keypoint_dir: Directory containing {video_id}.npy files.
    """
    keypoint_dir = Path(keypoint_dir)
    cache = {}

    attached = 0
    for rep in rep_segments:
        vid_id = rep["video_id"]

        if vid_id not in cache:
            npy_path = keypoint_dir / f"{vid_id}.npy"
            if not npy_path.exists():
                logger.warning("No keypoints for %s, skipping", vid_id)
                continue
            cache[vid_id] = np.load(npy_path)

        full_kps = cache[vid_id]
        start = rep["start_frame"]
        end = min(rep["end_frame"] + 1, len(full_kps))
        rep["keypoints"] = full_kps[start:end]
        attached += 1

    logger.info("Attached keypoints to %d / %d reps", attached, len(rep_segments))
