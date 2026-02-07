"""Sequence dataset for training the LSTM push-up phase detector."""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.counting.state_machine import PushUpPhase, PushUpStateMachine
from src.counting.lstm_counter import PushUpLSTM
from src.features.angles import compute_angle_sequence


class PushUpSequenceDataset(Dataset):
    """Sliding-window dataset over keypoint sequences with auto-generated labels.

    Labels are generated using the state machine (semi-supervised approach).

    Args:
        kps_sequences: List of (T_i, 12, 3) arrays — one per video.
        window_size: Number of frames per training sample.
        stride: Step size for the sliding window.
        down_threshold: State machine down threshold (degrees).
        up_threshold: State machine up threshold (degrees).
    """

    def __init__(
        self,
        kps_sequences: list[np.ndarray],
        window_size: int = 64,
        stride: int = 16,
        down_threshold: float = 90.0,
        up_threshold: float = 160.0,
    ):
        self.window_size = window_size
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []

        sm = PushUpStateMachine(
            down_threshold=down_threshold,
            up_threshold=up_threshold,
        )

        for kps_seq in kps_sequences:
            # Compute angle features: (T, 4)
            angles = compute_angle_sequence(kps_seq)

            # Generate phase labels using state machine
            phase_labels = sm.label_sequence(kps_seq)
            sm.reset()
            label_indices = np.array(
                [PushUpLSTM.PHASE_TO_IDX[p] for p in phase_labels],
                dtype=np.int64,
            )

            # Create sliding windows
            T = len(angles)
            for start in range(0, T - window_size + 1, stride):
                end = start + window_size
                self.samples.append((
                    angles[start:end],
                    label_indices[start:end],
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, labels = self.samples[idx]
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(labels).long(),
        )
