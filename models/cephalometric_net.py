"""Complete End-to-End Cephalometric Analysis Neural Network Architecture.

Assembles ImageEnhancement, EfficientNetV2Backbone, MultiScaleAFEM, CFFM, ACAM, GRLM,
and ClassificationHead into a unified, modular PyTorch deep learning model for
cephalometric X-ray classification. Supports Baseline-0 and Ablation study configurations.
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config
from backbone.efficientnet_v2 import EfficientNetV2Backbone
from modules.image_enhancement import ImageEnhancement
from modules.afem import MultiScaleAFEM
from modules.cffm import CFFM
from modules.acam import ACAM
from modules.grlm import GRLM
from models.classification_head import ClassificationHead


class CephalometricAnalysisNet(nn.Module):
    """End-to-End Modular Deep Learning Network for Cephalometric X-ray Analysis.

    Architecture Flow:
        Input Radiograph (B, 3, H, W)
            ↓
        EfficientNetV2Backbone ──► {P2, P3, P4, P5} (256 channels each)
            ↓
        MultiScaleAFEM (Optional) ► {P2', P3', P4', P5'}
            ↓
        CFFM (Optional) ─────────► Fused Feature Map
            ↓
        ACAM (Optional) ─────────► Context Aggregated Map
            ↓
        GRLM (Optional) ─────────► 512-D Embedding Vector
            ↓
        Classification Head ─────► Raw Logits (B, N_classes)
    """

    def __init__(self, config: Config = Config()) -> None:
        """Initialize complete CephalometricAnalysisNet pipeline.

        Args:
            config: Master Config instance containing sub-configurations for all modules.
        """
        super().__init__()
        self.config: Config = config

        # Preprocessing image enhancer
        self.image_enhancer = ImageEnhancement(config=self.config.image_enhancement)

        # Module 2: Backbone Feature Extractor (EfficientNetV2-S)
        self.backbone = EfficientNetV2Backbone(config=self.config.backbone)

        # Ablation Module Instantiations
        self.use_afem = self.config.ablation.use_afem
        self.use_cffm = self.config.ablation.use_cffm
        self.use_acam = self.config.ablation.use_acam
        self.use_grlm = self.config.ablation.use_grlm

        if self.use_afem:
            self.afem = MultiScaleAFEM(config=self.config.afem)
        else:
            self.afem = None

        if self.use_cffm:
            self.cffm = CFFM(config=self.config.cffm)
        else:
            self.cffm = None
            # Fallback 1x1 projection for simple concatenation
            self.fallback_cffm_proj = nn.Conv2d(256 * 4, 256, kernel_size=1, bias=False)

        if self.use_acam:
            self.acam = ACAM(config=self.config.acam)
        else:
            self.acam = None

        if self.use_grlm:
            self.grlm = GRLM(config=self.config.grlm)
            head_in_features = self.config.grlm.embedding_dim
        else:
            self.grlm = None
            # Fallback direct GAP if GRLM disabled
            head_in_features = 256

        # Classification Head (Dropout 0.3 + Linear -> Raw Logits)
        self.head_config = self.config.head
        self.head_config.in_features = head_in_features
        self.head = ClassificationHead(config=self.head_config)

    def count_parameters(self) -> Dict[str, int]:
        """Compute parameter counts for backbone, proposed modules, and total model.

        Returns:
            Dictionary containing total_params, trainable_params, frozen_params, and backbone_params.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        backbone_p = sum(p.numel() for p in self.backbone.parameters())

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "backbone_parameters": backbone_p,
            "proposed_modules_parameters": total - backbone_p,
        }

    def preprocess_image(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Execute 16-step ImageEnhancement pipeline on a raw image file or array."""
        return self.image_enhancer.process(image_input)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embedding vector from input image tensor."""
        # Step 1: Backbone features P2, P3, P4, P5
        raw_features: Dict[str, torch.Tensor] = self.backbone(x)

        # Step 2: Adaptive feature enhancement (AFEM)
        if self.use_afem and self.afem is not None:
            features = self.afem(raw_features)
        else:
            features = raw_features

        # Step 3: Multi-scale fusion (CFFM or Fallback)
        if self.use_cffm and self.cffm is not None:
            fused_features = self.cffm(features)
        else:
            # Fallback simple spatial upsample and concat fusion
            p2 = features["P2"]
            target_sz = (p2.shape[2], p2.shape[3])
            p3_up = F.interpolate(features["P3"], size=target_sz, mode="bilinear", align_corners=False)
            p4_up = F.interpolate(features["P4"], size=target_sz, mode="bilinear", align_corners=False)
            p5_up = F.interpolate(features["P5"], size=target_sz, mode="bilinear", align_corners=False)
            concat_feat = torch.cat([p2, p3_up, p4_up, p5_up], dim=1)
            fused_features = self.fallback_cffm_proj(concat_feat)

        # Step 4: Context aggregation (ACAM or Skip)
        if self.use_acam and self.acam is not None:
            context_features = self.acam(fused_features)
        else:
            context_features = fused_features

        # Step 5: Global representation learning (GRLM or GAP)
        if self.use_grlm and self.grlm is not None:
            embedding = self.grlm(context_features)
        else:
            # Baseline-0 fallback: direct GAP -> 256-D vector
            gap = torch.mean(context_features, dim=(2, 3))
            embedding = gap

        return embedding

    def predict_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute normalized class posterior probabilities from raw logits (Inference only)."""
        return F.softmax(logits, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute end-to-end forward pass returning raw unnormalized logits."""
        embedding = self.extract_embedding(x)
        logits = self.head(embedding)
        return logits
