"""Real-time per-rep form classification for the live demo.

Uses a trained ST-GCN model to classify each completed push-up rep
as correct or incorrect based on the keypoint frames within that rep.
"""

from pathlib import Path

import numpy as np
import torch

from src.classification.stgcn import PushUpSTGCN
from src.counting.state_machine import PushUpPhase
from src.features.normalize import torso_normalize

MAX_REP_FRAMES = 64


class FormChecker:
    """Per-rep form classifier for real-time inference.

    Accumulates keypoint frames during a rep and classifies the rep
    when it completes (GOING_UP -> UP transition).

    Args:
        model_path: Path to a saved ST-GCN state dict (.pt file).
        max_frames: Max frames for model input (pad/truncate to this).
        device_str: Device to run inference on.
    """

    def __init__(
        self,
        model_path: str | Path,
        max_frames: int = MAX_REP_FRAMES,
        device_str: str = "cpu",
    ):
        self.max_frames = max_frames
        self.device = torch.device(device_str)

        # Load model
        self.model = PushUpSTGCN(in_channels=2, num_classes=2)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Per-rep buffer
        self._rep_buffer: list[np.ndarray] = []
        self._in_rep = False
        self._prev_phase: PushUpPhase | None = None
        self._rep_count = 0
        self.last_result: tuple[str, float, int] | None = None

    def update(
        self, unified_kps: np.ndarray, phase: PushUpPhase,
    ) -> tuple[str, float, int] | None:
        """Add a keypoint frame and classify on rep completion.

        Args:
            unified_kps: (12, 3) unified keypoints for this frame.
            phase: Current phase from the state machine.

        Returns:
            (label, confidence, rep_number) when a rep just completed,
            or the most recent result otherwise.
            Returns None if no rep has completed yet.
        """
        # Normalize and center
        kps = torso_normalize(unified_kps)
        hip_mid = (kps[6, :2] + kps[7, :2]) / 2.0
        kps[:, :2] -= hip_mid
        xy = kps[:, :2]  # (12, 2)

        # Detect rep start: UP -> GOING_DOWN
        if (
            self._prev_phase == PushUpPhase.UP
            and phase == PushUpPhase.GOING_DOWN
        ):
            self._rep_buffer.clear()
            self._in_rep = True

        # Accumulate frames during a rep
        if self._in_rep:
            self._rep_buffer.append(xy)

        # Detect rep completion: GOING_UP -> UP
        if (
            self._prev_phase == PushUpPhase.GOING_UP
            and phase == PushUpPhase.UP
            and len(self._rep_buffer) >= 5
        ):
            self._rep_count += 1
            self.last_result = self._classify(self._rep_count)
            self._in_rep = False
            self._rep_buffer.clear()

        self._prev_phase = phase
        return self.last_result

    def _classify(self, rep_number: int) -> tuple[str, float, int]:
        """Run ST-GCN on the current rep buffer."""
        frames = np.stack(self._rep_buffer)  # (T, 12, 2)
        T = len(frames)

        # Pad or truncate to max_frames
        if T < self.max_frames:
            pad = np.zeros((self.max_frames - T, 12, 2), dtype=np.float32)
            frames = np.concatenate([frames, pad], axis=0)
        else:
            frames = frames[:self.max_frames]

        # (max_frames, 12, 2) -> (2, max_frames, 12)
        tensor = torch.from_numpy(frames.astype(np.float32)).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).to(self.device)  # (1, 2, T, 12)

        with torch.no_grad():
            logits = self.model(tensor)  # (1, 2)
            probs = torch.softmax(logits, dim=-1).squeeze()  # (2,)

        correct_prob = probs[0].item()
        incorrect_prob = probs[1].item()

        if correct_prob >= incorrect_prob:
            return ("correct", correct_prob, rep_number)
        else:
            return ("incorrect", incorrect_prob, rep_number)

    def reset(self) -> None:
        """Clear the buffer and reset state."""
        self._rep_buffer.clear()
        self._in_rep = False
        self._prev_phase = None
        self._rep_count = 0
        self.last_result = None
