"""Training loop for the LSTM push-up phase detector."""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import LeaveOneGroupOut

from src.counting.lstm_counter import PushUpLSTM
from src.counting.dataset import PushUpSequenceDataset

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")


def train_one_epoch(
    model: PushUpLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        log_probs = model(features)  # (B, T, 4)
        loss = criterion(log_probs.reshape(-1, 4), labels.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: PushUpLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model, return (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            log_probs = model(features)
            loss = criterion(log_probs.reshape(-1, 4), labels.reshape(-1))
            total_loss += loss.item()

            preds = log_probs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    n_batches = max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return total_loss / n_batches, accuracy


def train_with_cross_validation(
    kps_sequences: list[np.ndarray],
    video_groups: list[int] | None = None,
    window_size: int = 64,
    stride: int = 16,
    n_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_size: int = 64,
    device_str: str = "cpu",
) -> dict:
    """Train LSTM with leave-one-group-out cross-validation.

    Args:
        kps_sequences: List of (T_i, 12, 3) arrays.
        video_groups: Group ID for each sequence (for CV splits). If None, uses index.
        window_size: Sliding window size.
        stride: Sliding window stride.
        n_epochs: Training epochs per fold.
        batch_size: Batch size.
        lr: Learning rate.
        hidden_size: LSTM hidden size.
        device_str: Device string.

    Returns:
        Dict with per-fold results and best model state dict.
    """
    device = torch.device(device_str)

    # Build full dataset
    dataset = PushUpSequenceDataset(
        kps_sequences,
        window_size=window_size,
        stride=stride,
    )

    if video_groups is None:
        video_groups = list(range(len(kps_sequences)))

    # Assign group to each window sample
    sample_groups = []
    for vid_idx, kps_seq in enumerate(kps_sequences):
        T = len(kps_seq)
        n_windows = max(0, (T - window_size) // stride + 1)
        sample_groups.extend([video_groups[vid_idx]] * n_windows)
    sample_groups = np.array(sample_groups)

    logo = LeaveOneGroupOut()
    fold_results = []
    best_acc = 0.0
    best_state = None

    for fold, (train_idx, val_idx) in enumerate(logo.split(range(len(dataset)), groups=sample_groups)):
        logger.info("Fold %d: train=%d, val=%d samples", fold, len(train_idx), len(val_idx))

        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=batch_size
        )

        model = PushUpLSTM(input_size=4, hidden_size=hidden_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.NLLLoss()

        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            if epoch == n_epochs - 1:
                logger.info(
                    "Fold %d final — train_loss=%.4f, val_loss=%.4f, val_acc=%.4f",
                    fold, train_loss, val_loss, val_acc,
                )

        fold_results.append({"fold": fold, "val_loss": val_loss, "val_accuracy": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()

    # Save best model
    if best_state is not None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        save_path = MODEL_DIR / "lstm_counter_best.pt"
        torch.save(best_state, save_path)
        logger.info("Saved best model (acc=%.4f) to %s", best_acc, save_path)

    return {
        "fold_results": fold_results,
        "best_accuracy": best_acc,
    }
