"""YOLO11s-pose wrapper implementing the unified pose estimation interface."""

import logging
import time

import numpy as np
import torch
from ultralytics import YOLO

from src.pose_estimation.base import PoseEstimatorBase, PoseResult
from src.pose_estimation.keypoint_schema import (
    NUM_UNIFIED_JOINTS,
    YOLO_TO_UNIFIED,
    map_to_unified,
)

logger = logging.getLogger(__name__)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloEstimator(PoseEstimatorBase):
    """YOLO Pose estimator (YOLO11s-pose or YOLOv8s-pose fallback).

    Args:
        model_path: Path or name of the YOLO pose model.
        device: Device string or None for auto-detect.
        confidence: Minimum detection confidence.
    """

    DEFAULT_MODEL = "yolo11s-pose.pt"
    FALLBACK_MODEL = "yolov8s-pose.pt"

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        confidence: float = 0.5,
    ):
        self.device = device or _select_device()
        self.confidence = confidence

        model_path = model_path or self.DEFAULT_MODEL
        try:
            self._model = YOLO(model_path)
            self._model_name = model_path
        except Exception:
            logger.warning(
                "Failed to load %s, falling back to %s",
                model_path,
                self.FALLBACK_MODEL,
            )
            self._model = YOLO(self.FALLBACK_MODEL)
            self._model_name = self.FALLBACK_MODEL

    @property
    def model_name(self) -> str:
        return f"YOLO ({self._model_name})"

    @property
    def native_keypoint_count(self) -> int:
        return 17

    def predict_frame(self, frame: np.ndarray) -> PoseResult:
        t0 = time.perf_counter()
        results = self._model(
            frame, device=self.device, conf=self.confidence, verbose=False
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = results[0]
        native_kps = np.zeros((17, 3), dtype=np.float32)
        detected = False
        det_conf = 0.0

        if result.keypoints is not None and len(result.keypoints):
            kps_data = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
            scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.zeros(len(kps_data))
            best_idx = int(np.argmax(scores))
            native_kps = kps_data[best_idx]
            detected = True
            det_conf = float(scores[best_idx])

        unified_kps = map_to_unified(native_kps, YOLO_TO_UNIFIED)

        return PoseResult(
            native_keypoints=native_kps,
            unified_keypoints=unified_kps,
            detected=detected,
            detection_confidence=det_conf,
            inference_time_ms=elapsed_ms,
            model_name=self.model_name,
        )
