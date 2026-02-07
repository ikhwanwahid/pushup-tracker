"""LSTM-based push-up phase detector for repetition counting."""

import torch
import torch.nn as nn

from src.counting.state_machine import PushUpPhase


class PushUpLSTM(nn.Module):
    """Bidirectional LSTM for frame-level push-up phase classification.

    Input: (batch, seq_len, features)
    Output: (batch, seq_len, 4) — log-probabilities for 4 phases

    Rep count = number of GOING_UP -> UP transitions in predicted sequence.

    Args:
        input_size: Number of input features per frame.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between LSTM layers.
    """

    PHASE_TO_IDX = {
        PushUpPhase.UP: 0,
        PushUpPhase.GOING_DOWN: 1,
        PushUpPhase.DOWN: 2,
        PushUpPhase.GOING_UP: 3,
    }
    IDX_TO_PHASE = {v: k for k, v in PHASE_TO_IDX.items()}

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            (batch, seq_len, 4) log-probabilities.
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        logits = self.classifier(lstm_out)  # (batch, seq_len, 4)
        return torch.log_softmax(logits, dim=-1)

    @staticmethod
    def count_reps_from_predictions(phase_indices: list[int]) -> int:
        """Count reps from a sequence of predicted phase indices.

        A rep is counted each time GOING_UP (3) transitions to UP (0).
        """
        count = 0
        for i in range(1, len(phase_indices)):
            if phase_indices[i] == 0 and phase_indices[i - 1] == 3:
                count += 1
        return count
