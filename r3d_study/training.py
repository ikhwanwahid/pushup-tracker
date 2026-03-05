"""Training infrastructure for R3D-18 study.

Provides train/evaluate loops and k-fold cross-validation
that splits by video to prevent data leakage.

Supports separate train/val dataset factories for augmentation.
"""

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Evaluate model, return (loss, accuracy, predictions, true_labels)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n_batches = max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return total_loss / n_batches, accuracy, all_preds, all_labels


def run_rep_kfold_cv(
    model_factory: Callable[[], nn.Module],
    dataset_factory: Callable[[list[dict]], torch.utils.data.Dataset],
    rep_segments: list[dict],
    n_splits: int = 5,
    n_epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    patience: int = 15,
    device_str: str = "cpu",
    random_state: int = 42,
    train_dataset_factory: Callable[[list[dict]], torch.utils.data.Dataset] | None = None,
) -> dict:
    """Run per-rep k-fold CV, splitting by video to prevent data leakage.

    Args:
        model_factory: Callable returning a fresh model instance.
        dataset_factory: Callable taking rep dicts, returning a Dataset (used for val).
        rep_segments: List of rep dicts (must have "video_id" and "label" keys).
        n_splits: Number of CV folds.
        n_epochs: Maximum training epochs per fold.
        batch_size: Batch size.
        lr: Learning rate.
        patience: Early stopping patience.
        device_str: Device string ("cpu", "cuda", "mps").
        random_state: Random seed.
        train_dataset_factory: Optional separate factory for training (with augmentation).
            If None, uses dataset_factory for both train and val.

    Returns:
        Dict with fold_results, per_rep_preds, per_rep_true, best_state, best_accuracy.
    """
    device = torch.device(device_str)
    _train_factory = train_dataset_factory or dataset_factory

    # Get unique videos and their labels for stratified splitting
    video_ids = []
    video_labels = []
    seen = set()
    for rep in rep_segments:
        vid = rep["video_id"]
        if vid not in seen:
            video_ids.append(vid)
            video_labels.append(rep["label"])
            seen.add(vid)

    video_ids = np.array(video_ids)
    video_labels = np.array(video_labels)

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state,
    )

    per_rep_preds = [None] * len(rep_segments)
    per_rep_true = [rep["label"] for rep in rep_segments]

    fold_results = []
    best_overall_acc = 0.0
    best_state = None

    for fold, (train_vid_idx, val_vid_idx) in enumerate(
        skf.split(video_ids, video_labels)
    ):
        train_vid_set = set(video_ids[train_vid_idx])
        val_vid_set = set(video_ids[val_vid_idx])

        train_reps = [r for r in rep_segments if r["video_id"] in train_vid_set]
        val_reps = [r for r in rep_segments if r["video_id"] in val_vid_set]
        val_rep_indices = [
            i for i, r in enumerate(rep_segments) if r["video_id"] in val_vid_set
        ]

        logger.info(
            "Fold %d: train=%d reps (%d vids), val=%d reps (%d vids)",
            fold, len(train_reps), len(train_vid_set),
            len(val_reps), len(val_vid_set),
        )

        train_dataset = _train_factory(train_reps)   # with augmentation
        val_dataset = dataset_factory(val_reps)       # no augmentation

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        )

        model = model_factory().to(device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_epoch_state = None
        epochs_without_improvement = 0

        for epoch in range(n_epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
            )
            val_loss, val_acc, _, _ = evaluate(
                model, val_loader, criterion, device,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info(
                    "Fold %d: early stopping at epoch %d (best_acc=%.4f)",
                    fold, epoch, best_val_acc,
                )
                break

        # Load best model for this fold
        if best_epoch_state is not None:
            model.load_state_dict(best_epoch_state)
            model.to(device)

        _, final_acc, final_preds, _ = evaluate(
            model, val_loader, criterion, device,
        )

        for i, rep_idx in enumerate(val_rep_indices):
            per_rep_preds[rep_idx] = final_preds[i]

        fold_results.append({
            "fold": fold,
            "val_loss": val_loss,
            "val_accuracy": final_acc,
            "n_train_reps": len(train_reps),
            "n_val_reps": len(val_reps),
        })

        logger.info("Fold %d: val_accuracy=%.4f", fold, final_acc)

        if final_acc > best_overall_acc:
            best_overall_acc = final_acc
            best_state = best_epoch_state

    return {
        "fold_results": fold_results,
        "per_rep_preds": per_rep_preds,
        "per_rep_true": per_rep_true,
        "best_state": best_state,
        "best_accuracy": best_overall_acc,
    }
