"""R3D-18 3D CNN for binary push-up form classification.

Pretrained on Kinetics-400, supports frozen and unfrozen backbone configurations.
"""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class PushUpR3D(nn.Module):
    """R3D-18 fine-tuned for binary push-up form classification.

    Input: (B, 3, 16, 112, 112) — RGB video clips
    Output: (B, 2) — logits for [correct, incorrect]

    Args:
        freeze_backbone: If True, freeze all layers except the FC head.
        num_classes: Number of output classes.
    """

    # Block names in order from shallowest to deepest
    BLOCK_NAMES = ["layer1", "layer2", "layer3", "layer4"]

    def __init__(self, freeze_backbone: bool = True, num_classes: int = 2):
        super().__init__()

        self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

        # Replace the classifier head
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            self._freeze_all_except_fc()

    def _freeze_all_except_fc(self) -> None:
        """Freeze everything except the FC head."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze the last n residual blocks of the backbone.

        R3D-18 has 4 blocks: layer1, layer2, layer3, layer4.
        n=1 unfreezes layer4, n=2 unfreezes layer3+layer4, etc.

        Args:
            n: Number of blocks to unfreeze (1-4).
        """
        # Start frozen, then selectively unfreeze
        self._freeze_all_except_fc()

        blocks_to_unfreeze = self.BLOCK_NAMES[-n:]  # last n blocks
        for name, param in self.backbone.named_parameters():
            layer_name = name.split(".")[0]
            if layer_name in blocks_to_unfreeze:
                param.requires_grad = True

    def trainable_param_count(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, T, H, W) video tensor.

        Returns:
            (B, num_classes) logits.
        """
        return self.backbone(x)
