"""Abstract base class and data structures for pose estimators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class PoseResult:
    """Result from a single-frame pose estimation.

    Attributes:
        native_keypoints: Raw keypoints in the model's native format.
        unified_keypoints: Keypoints mapped to the 12-joint unified format, shape (12, 3).
        detected: Whether a person was detected in the frame.
        detection_confidence: Overall detection confidence score.
        inference_time_ms: Time taken for inference in milliseconds.
        model_name: Name of the model that produced this result.
    """

    native_keypoints: np.ndarray
    unified_keypoints: np.ndarray  # (12, 3) — [x, y, confidence]
    detected: bool
    detection_confidence: float
    inference_time_ms: float
    model_name: str


class PoseEstimatorBase(ABC):
    """Abstract base class for pose estimation models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name."""

    @property
    @abstractmethod
    def native_keypoint_count(self) -> int:
        """Number of keypoints in the model's native format."""

    @abstractmethod
    def predict_frame(self, frame: np.ndarray) -> PoseResult:
        """Run pose estimation on a single BGR frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            PoseResult with native and unified keypoints.
        """

    def predict_video(
        self,
        video_path: str,
        max_frames: int | None = None,
    ) -> list[PoseResult]:
        """Run pose estimation on all frames of a video.

        Args:
            video_path: Path to video file.
            max_frames: Stop after this many frames (None = all).

        Returns:
            List of PoseResult, one per frame.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None:
            total = min(total, max_frames)

        results = []
        for _ in tqdm(range(total), desc=f"{self.model_name} inference"):
            ret, frame = cap.read()
            if not ret:
                break
            results.append(self.predict_frame(frame))

        cap.release()
        return results
