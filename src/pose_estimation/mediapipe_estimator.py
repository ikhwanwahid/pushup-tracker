"""MediaPipe Pose Landmarker wrapper using the Tasks API.

Auto-downloads the .task model bundle from Google Cloud Storage.
Three complexity levels: lite, full, heavy.
"""

import logging
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks.python import BaseOptions, vision

from src.pose_estimation.base import PoseEstimatorBase, PoseResult
from src.pose_estimation.keypoint_schema import (
    MEDIAPIPE_TO_UNIFIED,
    map_to_unified,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")

_MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def _download_model(complexity: str) -> Path:
    """Download the PoseLandmarker .task bundle if not cached locally."""
    model_path = MODEL_DIR / f"pose_landmarker_{complexity}.task"
    if model_path.exists():
        return model_path

    url = _MODEL_URLS[complexity]
    logger.info("Downloading MediaPipe PoseLandmarker (%s) ...", complexity)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    model_path.write_bytes(resp.content)
    logger.info("Saved to %s (%d bytes)", model_path, len(resp.content))
    return model_path


class MediaPipeEstimator(PoseEstimatorBase):
    """MediaPipe PoseLandmarker estimator (Tasks API).

    Args:
        model_complexity: "lite", "full", or "heavy".
        min_detection_confidence: Minimum pose detection confidence.
        min_tracking_confidence: Minimum tracking confidence (VIDEO mode).
    """

    def __init__(
        self,
        model_complexity: str = "full",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if model_complexity not in _MODEL_URLS:
            raise ValueError(
                f"Unknown complexity '{model_complexity}'. Use 'lite', 'full', or 'heavy'."
            )
        self._complexity = model_complexity
        self._model_path = _download_model(model_complexity)
        self._min_det = min_detection_confidence
        self._min_track = min_tracking_confidence

        # Create IMAGE-mode landmarker for predict_frame
        self._landmarker = self._create_landmarker(vision.RunningMode.IMAGE)

    def _create_landmarker(self, mode: vision.RunningMode) -> vision.PoseLandmarker:
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=self._min_det,
            min_tracking_confidence=self._min_track,
        )
        return vision.PoseLandmarker.create_from_options(options)

    @property
    def model_name(self) -> str:
        return f"MediaPipe Pose ({self._complexity})"

    @property
    def native_keypoint_count(self) -> int:
        return 33

    def _extract_keypoints(
        self, result: vision.PoseLandmarkerResult, h: int, w: int
    ) -> tuple[np.ndarray, bool, float]:
        """Convert PoseLandmarkerResult to native keypoints array."""
        native_kps = np.zeros((33, 3), dtype=np.float32)
        detected = False
        det_conf = 0.0

        if result.pose_landmarks:
            detected = True
            landmarks = result.pose_landmarks[0]  # first (only) person
            confs = []
            for i, lm in enumerate(landmarks):
                native_kps[i, 0] = lm.x * w
                native_kps[i, 1] = lm.y * h
                native_kps[i, 2] = lm.visibility
                confs.append(lm.visibility)
            det_conf = float(np.mean(confs)) if confs else 0.0

        return native_kps, detected, det_conf

    def predict_frame(self, frame: np.ndarray) -> PoseResult:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        t0 = time.perf_counter()
        result = self._landmarker.detect(image)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        native_kps, detected, det_conf = self._extract_keypoints(result, h, w)
        unified_kps = map_to_unified(native_kps, MEDIAPIPE_TO_UNIFIED)

        return PoseResult(
            native_keypoints=native_kps,
            unified_keypoints=unified_kps,
            detected=detected,
            detection_confidence=det_conf,
            inference_time_ms=elapsed_ms,
            model_name=self.model_name,
        )

    def predict_video(
        self,
        video_path: str,
        max_frames: int | None = None,
    ) -> list[PoseResult]:
        """Run pose estimation using VIDEO mode (enables cross-frame tracking)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None:
            total = min(total, max_frames)

        # Use a separate VIDEO-mode landmarker
        video_landmarker = self._create_landmarker(vision.RunningMode.VIDEO)

        results = []
        frame_idx = 0
        from tqdm import tqdm

        for _ in tqdm(range(total), desc=f"{self.model_name} inference"):
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * (1000.0 / fps))

            t0 = time.perf_counter()
            mp_result = video_landmarker.detect_for_video(image, timestamp_ms)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            native_kps, detected, det_conf = self._extract_keypoints(mp_result, h, w)
            unified_kps = map_to_unified(native_kps, MEDIAPIPE_TO_UNIFIED)

            results.append(PoseResult(
                native_keypoints=native_kps,
                unified_keypoints=unified_kps,
                detected=detected,
                detection_confidence=det_conf,
                inference_time_ms=elapsed_ms,
                model_name=self.model_name,
            ))
            frame_idx += 1

        cap.release()
        video_landmarker.close()
        return results

    def close(self):
        """Release resources."""
        if hasattr(self, "_landmarker"):
            self._landmarker.close()

    def __del__(self):
        self.close()
