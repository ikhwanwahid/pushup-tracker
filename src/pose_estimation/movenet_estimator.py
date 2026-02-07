"""MoveNet Lightning/Thunder wrapper via ONNX Runtime.

Auto-downloads the ONNX model from TF Hub if not present locally.
Two variants: Lightning (192x192, faster) and Thunder (256x256, more accurate).
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from src.pose_estimation.base import PoseEstimatorBase, PoseResult
from src.pose_estimation.keypoint_schema import (
    MOVENET_TO_UNIFIED,
    map_to_unified,
)

logger = logging.getLogger(__name__)

# TF Hub model URLs (SavedModel format — we convert or use ONNX)
MOVENET_URLS = {
    "lightning": (
        "https://tfhub.dev/google/movenet/singlepose/lightning/4?lite-format=tflite",
        192,
    ),
    "thunder": (
        "https://tfhub.dev/google/movenet/singlepose/thunder/4?lite-format=tflite",
        256,
    ),
}

# Local model directory
MODEL_DIR = Path("models")


def _download_tflite(variant: str) -> Path:
    """Download MoveNet TFLite model if not cached locally."""
    model_path = MODEL_DIR / f"movenet_{variant}.tflite"
    if model_path.exists():
        return model_path

    url, _ = MOVENET_URLS[variant]
    logger.info("Downloading MoveNet %s from TF Hub...", variant)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, allow_redirects=True, timeout=60)
    resp.raise_for_status()
    model_path.write_bytes(resp.content)
    logger.info("Saved MoveNet %s to %s", variant, model_path)
    return model_path


class MoveNetEstimator(PoseEstimatorBase):
    """MoveNet pose estimator via ONNX Runtime or TFLite.

    Supports Lightning (fast) and Thunder (accurate) variants.

    Args:
        variant: "lightning" or "thunder".
        model_path: Explicit path to .tflite model (auto-downloads if None).
    """

    def __init__(
        self,
        variant: str = "lightning",
        model_path: str | None = None,
    ):
        if variant not in MOVENET_URLS:
            raise ValueError(f"Unknown variant: {variant}. Use 'lightning' or 'thunder'.")

        self._variant = variant
        _, self._input_size = MOVENET_URLS[variant]

        if model_path is not None:
            self._model_path = Path(model_path)
        else:
            self._model_path = _download_tflite(variant)

        self._interpreter = self._load_interpreter()

    def _load_interpreter(self):
        """Load TFLite interpreter, trying tflite-runtime first, then full TF."""
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=str(self._model_path))
        except ImportError:
            try:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=str(self._model_path))
            except ImportError:
                raise ImportError(
                    "Neither tflite-runtime nor tensorflow is installed. "
                    "Install one of: pip install tflite-runtime, pip install tensorflow"
                )
        interpreter.allocate_tensors()
        return interpreter

    @property
    def model_name(self) -> str:
        return f"MoveNet {self._variant.capitalize()}"

    @property
    def native_keypoint_count(self) -> int:
        return 17  # COCO-17

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and format frame for MoveNet input."""
        img = cv2.resize(frame, (self._input_size, self._input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0).astype(np.int32)
        return img

    def predict_frame(self, frame: np.ndarray) -> PoseResult:
        h, w = frame.shape[:2]
        input_tensor = self._preprocess(frame)

        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        self._interpreter.set_tensor(input_details[0]["index"], input_tensor)

        t0 = time.perf_counter()
        self._interpreter.invoke()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Output shape: (1, 1, 17, 3) — [y, x, confidence]
        raw_output = self._interpreter.get_tensor(output_details[0]["index"])
        keypoints_raw = raw_output[0, 0]  # (17, 3)

        # Convert from [y, x, conf] normalized to [x, y, conf] pixel coords
        native_kps = np.zeros((17, 3), dtype=np.float32)
        for i in range(17):
            native_kps[i, 0] = keypoints_raw[i, 1] * w  # x
            native_kps[i, 1] = keypoints_raw[i, 0] * h  # y
            native_kps[i, 2] = keypoints_raw[i, 2]       # confidence

        detected = bool(np.mean(native_kps[:, 2]) > 0.1)
        det_conf = float(np.mean(native_kps[:, 2]))

        unified_kps = map_to_unified(native_kps, MOVENET_TO_UNIFIED)

        return PoseResult(
            native_keypoints=native_kps,
            unified_keypoints=unified_kps,
            detected=detected,
            detection_confidence=det_conf,
            inference_time_ms=elapsed_ms,
            model_name=self.model_name,
        )
