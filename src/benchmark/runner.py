"""Multi-model benchmark harness for comparing pose estimation models."""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.pose_estimation.base import PoseEstimatorBase
from src.pose_estimation.keypoint_schema import UNIFIED_JOINTS

logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path("data/processed/benchmark")


class BenchmarkRunner:
    """Run multiple pose estimation models on the same set of videos and compare.

    Args:
        models: List of PoseEstimatorBase instances to benchmark.
        video_dir: Directory containing video files.
    """

    def __init__(
        self,
        models: list[PoseEstimatorBase],
        video_dir: str | Path = "data/raw/kaggle_pushups",
    ):
        self.models = models
        self.video_dir = Path(video_dir)

    def _find_videos(self) -> list[Path]:
        """Find all .mp4 files recursively."""
        return sorted(self.video_dir.rglob("*.mp4"))

    @staticmethod
    def _parse_label(video_path: Path) -> str:
        """Extract label from the video's parent directory name."""
        parent = video_path.parent.name.lower()
        if "correct" in parent and "wrong" not in parent:
            return "correct"
        return "incorrect"

    def run_full_benchmark(
        self,
        max_frames_per_video: int | None = None,
        max_videos: int | None = None,
        resume: bool = False,
    ) -> pd.DataFrame:
        """Run all models on all videos, collecting per-frame results.

        Args:
            max_frames_per_video: Stop after this many frames per video.
            max_videos: Limit the number of videos to process.
            resume: If True, load existing CSV and skip already-processed pairs.

        Returns:
            DataFrame with columns: model, video, label, frame, detected,
            inference_ms, detection_confidence, plus per-joint confidence columns.
        """
        videos = self._find_videos()
        if max_videos is not None:
            videos = videos[:max_videos]

        # Load existing results for resume
        csv_path = BENCHMARK_DIR / "full_benchmark.csv"
        existing_pairs: set[tuple[str, str]] = set()
        existing_df = None
        if resume and csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            existing_pairs = set(
                zip(existing_df["model"], existing_df["video"], strict=False)
            )
            logger.info("Resuming: %d (model, video) pairs already processed", len(existing_pairs))

        rows = []
        for model in self.models:
            logger.info("Benchmarking %s on %d videos", model.model_name, len(videos))
            for video_path in tqdm(
                videos, desc=model.model_name, unit="video"
            ):
                video_name = video_path.relative_to(self.video_dir).as_posix()
                label = self._parse_label(video_path)

                if (model.model_name, video_name) in existing_pairs:
                    continue

                results = model.predict_video(
                    str(video_path), max_frames=max_frames_per_video
                )
                for frame_idx, result in enumerate(results):
                    row = {
                        "model": result.model_name,
                        "video": video_name,
                        "label": label,
                        "frame": frame_idx,
                        "detected": result.detected,
                        "inference_ms": result.inference_time_ms,
                        "detection_confidence": result.detection_confidence,
                    }
                    for j, joint_name in enumerate(UNIFIED_JOINTS):
                        row[f"conf_{joint_name}"] = float(
                            result.unified_keypoints[j, 2]
                        )
                    rows.append(row)

        df = pd.DataFrame(rows)
        if existing_df is not None and not df.empty:
            df = pd.concat([existing_df, df], ignore_index=True)
        elif existing_df is not None:
            df = existing_df

        self._save(df, "full_benchmark.csv")
        return df

    def run_latency_benchmark(
        self,
        n_frames: int = 100,
        warmup_frames: int = 10,
    ) -> pd.DataFrame:
        """Measure inference latency per model on synthetic frames.

        Args:
            n_frames: Number of frames to time (after warmup).
            warmup_frames: Frames to run before timing starts.

        Returns:
            DataFrame with per-model latency statistics.
        """
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        rows = []

        for model in self.models:
            logger.info("Latency benchmark: %s", model.model_name)

            # Warmup
            for _ in tqdm(range(warmup_frames), desc=f"{model.model_name} warmup", leave=False):
                model.predict_frame(frame)

            # Timed runs
            times = []
            for _ in tqdm(range(n_frames), desc=f"{model.model_name} timing"):
                t0 = time.perf_counter()
                model.predict_frame(frame)
                times.append((time.perf_counter() - t0) * 1000.0)

            times = np.array(times)
            rows.append({
                "model": model.model_name,
                "mean_ms": float(np.mean(times)),
                "median_ms": float(np.median(times)),
                "std_ms": float(np.std(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
                "n_frames": n_frames,
            })

        df = pd.DataFrame(rows)
        self._save(df, "latency_benchmark.csv")
        return df

    def _save(self, df: pd.DataFrame, filename: str) -> None:
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        path = BENCHMARK_DIR / filename
        df.to_csv(path, index=False)
        logger.info("Saved benchmark results to %s", path)
