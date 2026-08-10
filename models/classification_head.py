"""Classification Head Module for Cephalometric Radiograph Analysis.

Maps normalized 512-dimensional GRLM embedding vectors to raw class logits
across cephalometric analysis categories (e.g., Skeletal Class I, II, III).
Softmax is deliberately excluded during forward pass for PyTorch CrossEntropyLoss compatibility.
"""

import torch
import torch.nn as nn

from configs.config import ClassificationHeadConfig


class ClassificationHead(nn.Module):
    """Classification Head Layer for Cephalometric Analysis.

    Dataflow:
        Global Feature Vector (512-D Embedding from GRLM)
            ↓
        Dropout(p = 0.3)
            ↓
        Fully Connected Layer (512 -> N_CLASSES)
            ↓
        Raw Logits
    """

    def __init__(self, config: ClassificationHeadConfig = ClassificationHeadConfig()) -> None:
        """Initialize Classification Head with configuration settings.

        Args:
            config: ClassificationHeadConfig object specifying input/output dimensions and dropout.
        """
        super().__init__()
        self.config: ClassificationHeadConfig = config

        self.dropout = nn.Dropout(p=self.config.dropout_rate)
        self.fc = nn.Linear(self.config.in_features, self.config.num_classes, bias=True)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Execute Classification Head forward pass returning raw unnormalized logits.

        Args:
            embedding: Normalized global feature embedding vector of shape (B, 512).

        Returns:
            Unnormalized raw class logits tensor of shape (B, N_classes).
        """
        # Step 1: Apply Dropout (0.3)
        dropped_embedding = self.dropout(embedding)

        # Step 2: Linear projection to raw class logits (B, N_classes)
        logits = self.fc(dropped_embedding)

        return logits
