"""3D CNN wrapper for push-up form classification.

Uses a pretrained R3D-18 (Kinetics-400) with a replaced FC head
for binary correct/incorrect classification.
"""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class PushUpVideoClassifier(nn.Module):
    """R3D-18 fine-tuned for binary push-up form classification.

    Freezes all backbone layers; only the final FC layer is trainable.

    Input: (B, 3, 16, 112, 112) — RGB video clips
    Output: (B, 2) — logits for [correct, incorrect]
    """

    def __init__(self, freeze_backbone: bool = True, num_classes: int = 2):
        super().__init__()

        self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

        # Replace the classifier head
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            # Freeze everything except the new FC layer
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, T, H, W) video tensor.

        Returns:
            (B, num_classes) logits
        """
        return self.backbone(x)
