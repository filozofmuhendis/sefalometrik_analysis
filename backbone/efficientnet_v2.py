"""EfficientNetV2-S Backbone Feature Extractor for Cephalometric X-ray Analysis.

Extracts multi-scale hierarchical feature maps P2, P3, P4, P5 from Stage 2, Stage 3,
Stage 4, and Stage 5 of ImageNet pretrained EfficientNetV2-S architecture.
Aligns channel dimensions of all stages to 256 channels via 1x1 convolutions.
Supports Progressive Fine-Tuning across three discrete training phases.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from configs.config import BackboneConfig


class EfficientNetV2Backbone(nn.Module):
    """Multi-Scale EfficientNetV2-S Feature Extractor with Channel Projection & Progressive Fine-Tuning.

    Pyramid Level Mapping:
        Stage 2 (stride 4,  48 channels)  -> 1x1 Conv -> P2 (256 channels)
        Stage 3 (stride 8,  64 channels)  -> 1x1 Conv -> P3 (256 channels)
        Stage 4 (stride 16, 128 channels) -> 1x1 Conv -> P4 (256 channels)
        Stage 5 (stride 32, 256 channels) -> 1x1 Conv -> P5 (256 channels)
    """

    def __init__(self, config: BackboneConfig = BackboneConfig()) -> None:
        """Initialize EfficientNetV2-S Backbone and 1x1 lateral projection modules.

        Args:
            config: BackboneConfig object specifying hyperparameters.
        """
        super().__init__()
        self.config: BackboneConfig = config

        # Load EfficientNetV2-S model (ImageNet pretrained or random init)
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if self.config.pretrained else None
        effnet = models.efficientnet_v2_s(weights=weights)

        # Isolate feature backbone stages 0 to 6 (Stage 7 head conv and classifier removed)
        self.features = effnet.features[:7]

        # Channel dimensions for Stage 2 (48), Stage 3 (64), Stage 4 (128), Stage 5 (256)
        in_ch_p2, in_ch_p3, in_ch_p4, in_ch_p5 = 48, 64, 128, 256
        out_ch: int = self.config.out_channels

        # 1x1 Convolution lateral projections to uniform 256 channels
        self.proj_p2 = nn.Sequential(
            nn.Conv2d(in_ch_p2, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.proj_p3 = nn.Sequential(
            nn.Conv2d(in_ch_p3, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.proj_p4 = nn.Sequential(
            nn.Conv2d(in_ch_p4, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.proj_p5 = nn.Sequential(
            nn.Conv2d(in_ch_p5, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

        # Configure progressive fine-tuning phase
        self.set_progressive_phase(phase=self.config.progressive_phase)

    def set_progressive_phase(self, phase: int) -> None:
        """Configure progressive fine-tuning parameter freezing across phases.

        Phase 1: Freeze Stem, Stage 1 (features[0,1]), Stage 2 (features[2]). Train Stage 3+ & Projections.
        Phase 2: Unfreeze Stage 2 (features[2]). Train Stage 2+ & Projections.
        Phase 3: Unfreeze all parameters (Full Fine-Tuning).

        Args:
            phase: Integer phase indicator (1, 2, or 3).

        Raises:
            ValueError: If phase is not in {1, 2, 3}.
        """
        if phase not in (1, 2, 3):
            raise ValueError(f"Invalid progressive phase: {phase}. Must be 1, 2, or 3.")

        self.config.progressive_phase = phase

        # Initially unfreeze all backbone parameters
        for param in self.features.parameters():
            param.requires_grad = True

        # Keep 1x1 projection layers trainable across all phases
        for proj in [self.proj_p2, self.proj_p3, self.proj_p4, self.proj_p5]:
            for param in proj.parameters():
                param.requires_grad = True

        if phase == 1:
            # Freeze Stem (features[0]), Stage 1 (features[1]), Stage 2 (features[2])
            for stage_idx in [0, 1, 2]:
                for param in self.features[stage_idx].parameters():
                    param.requires_grad = False

        elif phase == 2:
            # Freeze Stem (features[0]) and Stage 1 (features[1]), unfreeze Stage 2+
            for stage_idx in [0, 1]:
                for param in self.features[stage_idx].parameters():
                    param.requires_grad = False

        elif phase == 3:
            # Phase 3: All parameters unfrozen for full fine-tuning
            pass

    def get_trainable_parameters_count(self) -> Tuple[int, int]:
        """Compute trainable vs total parameters.

        Returns:
            Tuple of (trainable_params_count, total_params_count).
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale feature maps P2, P3, P4, P5.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Dictionary mapping stage labels 'P2', 'P3', 'P4', 'P5' to tensors of shape (B, 256, H_i, W_i).
        """
        # Sequential feature extraction through EfficientNetV2-S stages
        feat_stage2 = self.features[:3](x)       # Stage 2 (stride 4, 48 ch)
        feat_stage3 = self.features[3](feat_stage2)  # Stage 3 (stride 8, 64 ch)
        feat_stage4 = self.features[4](feat_stage3)  # Stage 4 (stride 16, 128 ch)
        feat_stage5 = self.features[5:7](feat_stage4) # Stage 5 (stride 32, 256 ch)

        # 1x1 Conv projections to uniform 256 channels
        p2 = self.proj_p2(feat_stage2)
        p3 = self.proj_p3(feat_stage3)
        p4 = self.proj_p4(feat_stage4)
        p5 = self.proj_p5(feat_stage5)

        return {
            "P2": p2,
            "P3": p3,
            "P4": p4,
            "P5": p5,
        }
