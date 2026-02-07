"""Extract keypoints from all videos using multiple pose estimation models.

Saves per-video keypoint arrays as .npy files and collects benchmark stats.

Usage:
    python -m src.benchmark.extract_keypoints --models yolo mediapipe
    python -m src.benchmark.extract_keypoints --models yolo --max-videos 2
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.pose_estimation.keypoint_schema import UNIFIED_JOINTS

logger = logging.getLogger(__name__)

VIDEO_DIR = Path("data/raw/kaggle_pushups")
KEYPOINTS_DIR = Path("data/processed/keypoints")
BENCHMARK_DIR = Path("data/processed/benchmark")

MODEL_REGISTRY = {
    "yolo": ("src.pose_estimation.yolo_estimator", "YoloEstimator", {}),
    "mediapipe": ("src.pose_estimation.mediapipe_estimator", "MediaPipeEstimator", {}),
}


def _make_model(name: str):
    """Lazily import and instantiate a model by registry name."""
    module_path, class_name, kwargs = MODEL_REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def _parse_label(video_path: Path) -> str:
    """Extract label (correct/incorrect) from the video's parent directory."""
    parent = video_path.parent.name.lower()
    if "correct" in parent and "wrong" not in parent:
        return "correct"
    return "incorrect"


def _sanitize_stem(video_path: Path) -> str:
    """Create a filesystem-safe stem from a video path."""
    return video_path.stem.replace(" ", "_").replace("(", "").replace(")", "")


def _find_videos() -> list[Path]:
    """Find all .mp4 files in the dataset directory."""
    return sorted(VIDEO_DIR.rglob("*.mp4"))


def extract_for_model(
    model_name: str,
    videos: list[Path],
    resume: bool = True,
) -> list[dict]:
    """Run a single model on all videos, saving keypoints and collecting stats.

    Returns list of per-frame stat dicts.
    """
    model = _make_model(model_name)
    model_dir = KEYPOINTS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for video_path in tqdm(videos, desc=f"{model_name}", unit="video"):
        label = _parse_label(video_path)
        stem = _sanitize_stem(video_path)
        out_path = model_dir / f"{label}__{stem}.npy"

        # Resume: skip if output already exists
        if resume and out_path.exists():
            logger.info("Skipping %s (already exists)", out_path)
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Cannot open %s, skipping", video_path)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        all_kps = []  # will be (T, 12, 3)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = model.predict_frame(frame)
            all_kps.append(result.unified_keypoints)

            row = {
                "model": model_name,
                "video": video_path.relative_to(VIDEO_DIR).as_posix(),
                "label": label,
                "frame": frame_idx,
                "detected": result.detected,
                "inference_ms": result.inference_time_ms,
                "detection_confidence": result.detection_confidence,
            }
            for j, joint_name in enumerate(UNIFIED_JOINTS):
                row[f"conf_{joint_name}"] = float(result.unified_keypoints[j, 2])
            rows.append(row)
            frame_idx += 1

        cap.release()

        # Save keypoints array
        if all_kps:
            kps_array = np.stack(all_kps, axis=0)  # (T, 12, 3)
            np.save(out_path, kps_array)
            logger.debug("Saved %s — shape %s", out_path, kps_array.shape)

    return rows


def save_manifest(videos: list[Path]) -> None:
    """Save a manifest.json mapping sanitized filenames to original metadata."""
    manifest = {}
    for video_path in videos:
        label = _parse_label(video_path)
        stem = _sanitize_stem(video_path)
        key = f"{label}__{stem}"

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        manifest[key] = {
            "original_path": str(video_path.relative_to(VIDEO_DIR)),
            "label": label,
            "fps": fps,
            "resolution": [w, h],
            "n_frames": n_frames,
            "duration_s": round(n_frames / fps, 2) if fps > 0 else 0,
        }

    out = KEYPOINTS_DIR / "manifest.json"
    KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    logger.info("Saved manifest with %d entries to %s", len(manifest), out)


def main():
    parser = argparse.ArgumentParser(description="Extract keypoints from videos")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=list(MODEL_REGISTRY.keys()),
        help="Which models to run",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit number of videos (for quick testing)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-process even if output files exist",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    videos = _find_videos()
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    logger.info("Found %d videos", len(videos))

    # Save manifest (always, it's fast)
    save_manifest(videos)

    # Run each model
    all_rows = []
    for model_name in args.models:
        logger.info("=== Extracting with %s ===", model_name)
        rows = extract_for_model(model_name, videos, resume=not args.no_resume)
        all_rows.extend(rows)

    # Save benchmark CSV
    if all_rows:
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_rows)

        csv_path = BENCHMARK_DIR / "full_benchmark.csv"
        if csv_path.exists() and not args.no_resume:
            existing = pd.read_csv(csv_path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["model", "video", "frame"], keep="last")

        df.to_csv(csv_path, index=False)
        logger.info("Saved %d rows to %s", len(df), csv_path)
    else:
        logger.info("No new data to save (all videos already processed)")


if __name__ == "__main__":
    main()
