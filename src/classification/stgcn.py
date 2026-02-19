"""Spatial-Temporal Graph Convolutional Network for push-up form classification.

Implements ST-GCN using standard PyTorch (no torch_geometric dependency).
Graph structure is derived from the unified 12-joint skeleton.
"""

import torch
import torch.nn as nn
import numpy as np

from src.pose_estimation.keypoint_schema import UNIFIED_SKELETON, NUM_UNIFIED_JOINTS


def build_adjacency_matrix() -> torch.Tensor:
    """Build normalized adjacency matrix from the unified skeleton.

    Uses symmetric normalization: D^{-1/2} (A+I) D^{-1/2}

    Returns:
        Tensor of shape (12, 12) — normalized adjacency matrix.
    """
    n = NUM_UNIFIED_JOINTS  # 12
    A = np.zeros((n, n), dtype=np.float32)

    for i, j in UNIFIED_SKELETON:
        A[i, j] = 1.0
        A[j, i] = 1.0

    # Add self-loops
    A += np.eye(n, dtype=np.float32)

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return torch.from_numpy(A_norm)


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
        # Graph convolution: aggregate neighbor features via adjacency
        # A is (V, V), x is (B, T, V, C) -> einsum contracts over V
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
        res = x.permute(0, 3, 1, 2).reshape(B * V, C, T)  # (B*V, C, T)
        res = self.residual(res)  # (B*V, out_ch, T')
        T_out = res.shape[-1]
        out_ch = res.shape[1]
        res = res.reshape(B, V, out_ch, T_out).permute(0, 2, 3, 1)  # (B, out_ch, T', V)

        # Spatial graph convolution
        x = x.permute(0, 2, 3, 1)  # (B, T, V, C)
        x = self.spatial(x)  # (B, T, V, out_ch)
        x = x.permute(0, 3, 1, 2)  # (B, out_ch, T, V)

        # BatchNorm over channels: reshape to (B*T*V, out_ch) -> BN -> reshape back
        x = x.permute(0, 2, 3, 1).reshape(-1, out_ch)
        x = self.bn_spatial(x.unsqueeze(-1)).squeeze(-1)
        x = x.reshape(B, T, V, out_ch).permute(0, 3, 1, 2)  # (B, out_ch, T, V)

        x = self.relu(x)

        # Temporal convolution: reshape to (B*V, out_ch, T)
        x = x.permute(0, 3, 1, 2).reshape(B * V, out_ch, T)  # (B*V, out_ch, T)
        x = self.temporal(x)  # (B*V, out_ch, T')
        x = x.reshape(B, V, out_ch, T_out).permute(0, 2, 3, 1)  # (B, out_ch, T', V)

        # BatchNorm
        x = x.permute(0, 2, 3, 1).reshape(-1, out_ch)
        x = self.bn_temporal(x.unsqueeze(-1)).squeeze(-1)
        x = x.reshape(B, T_out, V, out_ch).permute(0, 3, 1, 2)  # (B, out_ch, T', V)

        x = self.relu(x + res)
        x = self.dropout(x)
        return x


class PushUpSTGCN(nn.Module):
    """ST-GCN for binary push-up form classification.

    Input: (B, 2, T, 12) — (x,y) coordinates, T timesteps, 12 joints
    Output: (B, 2) — logits for [correct, incorrect]
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.block1 = STGCNBlock(in_channels, 64, stride=1, dropout=dropout)
        self.block2 = STGCNBlock(64, 64, stride=1, dropout=dropout)
        self.block3 = STGCNBlock(64, 128, stride=1, dropout=dropout)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 2, T, 12) — (x,y) coords over time for 12 joints.

        Returns:
            (B, num_classes) logits
        """
        x = self.block1(x)  # (B, 64, T, 12)
        x = self.block2(x)  # (B, 64, T, 12)
        x = self.block3(x)  # (B, 128, T, 12)

        # Global average pool over time and joints
        x = x.mean(dim=[2, 3])  # (B, 128)
        x = self.fc(x)  # (B, num_classes)
        return x
