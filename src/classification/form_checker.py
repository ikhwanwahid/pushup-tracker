"""Real-time form classification for the live demo.

Uses a trained ST-GCN model to classify push-up form as correct/incorrect
from a rolling buffer of keypoint frames.
"""

from pathlib import Path

import numpy as np
import torch

from src.classification.stgcn import PushUpSTGCN
from src.features.normalize import torso_normalize


class FormChecker:
    """Rolling-buffer form classifier for real-time inference.

    Accumulates keypoint frames in a buffer and periodically runs
    the ST-GCN model to classify form as correct or incorrect.

    Args:
        model_path: Path to a saved ST-GCN state dict (.pt file).
        buffer_size: Number of frames to keep in the rolling buffer.
        classify_interval: Run classification every N frames.
        device_str: Device to run inference on.
    """

    def __init__(
        self,
        model_path: str | Path,
        buffer_size: int = 150,
        classify_interval: int = 30,
        device_str: str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.classify_interval = classify_interval
        self.device = torch.device(device_str)

        # Load model
        self.model = PushUpSTGCN(in_channels=2, num_classes=2)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Rolling buffer for keypoint frames
        self.buffer: list[np.ndarray] = []
        self.frame_count = 0
        self.last_result: tuple[str, float] | None = None

    def update(self, unified_kps: np.ndarray) -> tuple[str, float] | None:
        """Add a keypoint frame and optionally classify.

        Args:
            unified_kps: (12, 3) unified keypoints for this frame.

        Returns:
            (label, confidence) if classification was run this frame,
            or the most recent result otherwise.
            label is "correct" or "incorrect", confidence is 0-1.
            Returns None if not enough frames have been accumulated.
        """
        # Normalize and center
        kps = torso_normalize(unified_kps)
        hip_mid = (kps[6, :2] + kps[7, :2]) / 2.0
        kps[:, :2] -= hip_mid
        xy = kps[:, :2]  # (12, 2)

        self.buffer.append(xy)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        self.frame_count += 1

        # Classify every N frames once we have enough data
        if self.frame_count % self.classify_interval == 0 and len(self.buffer) >= 30:
            self.last_result = self._classify()

        return self.last_result

    def _classify(self) -> tuple[str, float]:
        """Run ST-GCN on the current buffer."""
        frames = np.stack(self.buffer)  # (T, 12, 2)
        T = len(frames)

        # Pad to buffer_size if needed
        if T < self.buffer_size:
            pad = np.zeros((self.buffer_size - T, 12, 2), dtype=np.float32)
            frames = np.concatenate([frames, pad], axis=0)

        # (buffer_size, 12, 2) -> (2, buffer_size, 12)
        tensor = torch.from_numpy(frames.astype(np.float32)).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).to(self.device)  # (1, 2, T, 12)

        with torch.no_grad():
            logits = self.model(tensor)  # (1, 2)
            probs = torch.softmax(logits, dim=-1).squeeze()  # (2,)

        correct_prob = probs[0].item()
        incorrect_prob = probs[1].item()

        if correct_prob >= incorrect_prob:
            return ("correct", correct_prob)
        else:
            return ("incorrect", incorrect_prob)

    def reset(self) -> None:
        """Clear the buffer and reset state."""
        self.buffer.clear()
        self.frame_count = 0
        self.last_result = None
