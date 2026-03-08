"""Spatial-Temporal Graph Convolutional Network for push-up form classification.

Self-contained ST-GCN implementation using standard PyTorch (no torch_geometric).
Graph structure is derived from the unified 12-joint skeleton.

Supports configurable depth and width for architecture comparison.
"""

import torch
import torch.nn as nn
import numpy as np


# ============================================================
# Skeleton graph definition (unified 12-joint format)
# ============================================================

NUM_JOINTS = 12

# Skeleton connections (unified joint indices)
SKELETON_EDGES = [
    (0, 1),    # shoulder-shoulder
    (0, 2),    # left shoulder-elbow
    (2, 4),    # left elbow-wrist
    (1, 3),    # right shoulder-elbow
    (3, 5),    # right elbow-wrist
    (0, 6),    # left shoulder-hip
    (1, 7),    # right shoulder-hip
    (6, 7),    # hip-hip
    (6, 8),    # left hip-knee
    (8, 10),   # left knee-ankle
    (7, 9),    # right hip-knee
    (9, 11),   # right knee-ankle
]


def build_adjacency_matrix() -> torch.Tensor:
    """Build normalized adjacency matrix from the unified skeleton.

    Uses symmetric normalization: D^{-1/2} (A+I) D^{-1/2}

    Returns:
        Tensor of shape (12, 12) — normalized adjacency matrix.
    """
    n = NUM_JOINTS
    A = np.zeros((n, n), dtype=np.float32)

    for i, j in SKELETON_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0

    # Add self-loops
    A += np.eye(n, dtype=np.float32)

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return torch.from_numpy(A_norm)


# ============================================================
# ST-GCN building blocks
# ============================================================


class SpatialGraphConv(nn.Module):
    """Graph convolution on the spatial (joint) dimension."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.register_buffer("A", build_adjacency_matrix())
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, V, C) — batch, time, joints, channels.

        Returns:
            (B, T, V, out_channels)
        """
        x = torch.einsum("vu,btuc->btvc", self.A, x)
        x = self.linear(x)
        return x


class STGCNBlock(nn.Module):
    """One ST-GCN block: spatial graph conv + temporal conv + residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.spatial = SpatialGraphConv(in_channels, out_channels)
        self.bn_spatial = nn.BatchNorm1d(out_channels)

        # Temporal convolution: operates along the time axis
        self.temporal = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=9, stride=stride, padding=4,
        )
        self.bn_temporal = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, T, V) — batch, channels, time, joints.

        Returns:
            (B, out_channels, T', V)
        """
        B, C, T, V = x.shape

        # Residual: reshape to (B*V, C, T) for Conv1d
        res = x.permute(0, 3, 1, 2).reshape(B * V, C, T)
        res = self.residual(res)
        T_out = res.shape[-1]
        out_ch = res.shape[1]
        res = res.reshape(B, V, out_ch, T_out).permute(0, 2, 3, 1)

        # Spatial graph convolution
        x = x.permute(0, 2, 3, 1)  # (B, T, V, C)
        x = self.spatial(x)         # (B, T, V, out_ch)
        x = x.permute(0, 3, 1, 2)  # (B, out_ch, T, V)

        # BatchNorm over channels
        x = x.permute(0, 2, 3, 1).reshape(-1, out_ch)
        x = self.bn_spatial(x.unsqueeze(-1)).squeeze(-1)
        x = x.reshape(B, T, V, out_ch).permute(0, 3, 1, 2)

        x = self.relu(x)

        # Temporal convolution
        x = x.permute(0, 3, 1, 2).reshape(B * V, out_ch, T)
        x = self.temporal(x)
        x = x.reshape(B, V, out_ch, T_out).permute(0, 2, 3, 1)

        # BatchNorm
        x = x.permute(0, 2, 3, 1).reshape(-1, out_ch)
        x = self.bn_temporal(x.unsqueeze(-1)).squeeze(-1)
        x = x.reshape(B, T_out, V, out_ch).permute(0, 3, 1, 2)

        x = self.relu(x + res)
        x = self.dropout(x)
        return x


# ============================================================
# Full ST-GCN model
# ============================================================

# Predefined channel configurations
CONFIGS = {
    "small":  [32, 64],
    "medium": [64, 64, 128],
    "large":  [64, 128, 128, 256],
}


class PushUpSTGCN(nn.Module):
    """ST-GCN for binary push-up form classification.

    Input: (B, in_channels, T, 12) — coordinates over T timesteps for 12 joints.
    Output: (B, 2) — logits for [correct, incorrect]

    Args:
        in_channels: Number of input channels per joint (2=xy, 3=xy+confidence).
        num_classes: Number of output classes.
        channels: List of output channels for each ST-GCN block.
            Use CONFIGS["small"], CONFIGS["medium"], or CONFIGS["large"].
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 2,
        channels: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        if channels is None:
            channels = CONFIGS["medium"]

        self.channels = channels
        self.in_channels = in_channels

        # Build ST-GCN blocks
        blocks = []
        prev_ch = in_channels
        for ch in channels:
            blocks.append(STGCNBlock(prev_ch, ch, stride=1, dropout=dropout))
            prev_ch = ch
        self.blocks = nn.ModuleList(blocks)

        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, in_channels, T, 12)

        Returns:
            (B, num_classes) logits
        """
        for block in self.blocks:
            x = block(x)

        # Global average pool over time and joints
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return x

    def trainable_param_count(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_name(self) -> str:
        """Return a descriptive name for the channel configuration."""
        for name, ch_list in CONFIGS.items():
            if self.channels == ch_list:
                return name
        return f"custom_{len(self.channels)}blocks"
