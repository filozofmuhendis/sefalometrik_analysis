"""Class Weighted Cross Entropy Loss Module.

Computes Balanced Class Weighted Cross Entropy Loss to address severe class imbalance
in cephalometric radiograph datasets.

Formula:
    weight_i = N / (C * n_i)
where:
    N = total training samples
    C = total number of classes
    n_i = number of training samples in class i
"""

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassWeightedCrossEntropyLoss(nn.Module):
    """Balanced Class Weighted Cross Entropy Loss wrapper."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None) -> None:
        """Initialize ClassWeightedCrossEntropyLoss.

        Args:
            class_weights: Optional 1D float32 tensor of class weight factors.
        """
        super().__init__()
        if class_weights is not None:
            self.class_weights: Optional[torch.Tensor] = class_weights.float()
        else:
            self.class_weights = None

    @staticmethod
    def compute_balanced_class_weights(
        train_targets: Union[List[int], np.ndarray, torch.Tensor],
        num_classes: int,
    ) -> torch.Tensor:
        """Compute balanced class weights strictly from training set targets.

        Formula: weight_i = N / (C * n_i)

        Args:
            train_targets: Training set target label collection.
            num_classes: Total number of classes (C).

        Returns:
            Float32 torch.Tensor of balanced class weights of shape (num_classes,).
        """
        if isinstance(train_targets, (list, tuple)):
            targets_tensor = torch.tensor(train_targets, dtype=torch.long)
        elif isinstance(train_targets, np.ndarray):
            targets_tensor = torch.from_numpy(train_targets).long()
        else:
            targets_tensor = train_targets.long()

        total_samples: int = len(targets_tensor)
        counts = torch.bincount(targets_tensor, minlength=num_classes).float()

        # Handle potential empty classes gracefully to avoid division by zero
        safe_counts = torch.clamp(counts, min=1.0)

        # Balanced formula: weight_i = N / (C * n_i)
        weights = total_samples / (num_classes * safe_counts)
        return weights.float()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate weighted cross entropy loss on raw logits.

        Args:
            logits: Unnormalized raw class logits tensor of shape (B, N_classes).
            targets: Ground truth target class index tensor of shape (B,).

        Returns:
            Scalar loss tensor.
        """
        if self.class_weights is not None:
            weights = self.class_weights.to(device=logits.device, dtype=torch.float32)
        else:
            weights = None

        return F.cross_entropy(logits, targets, weight=weights)
