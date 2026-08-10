"""Structured Prediction Results for Cephalometric Radiograph Inference.

Defines the PredictionResult class representing the structured outputs of
the inference pipeline, including class index, name, probabilities, confidence,
and optional intermediate embeddings or preprocessed images.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass(frozen=True)
class PredictionResult:
    """Structured result container for Cephalometric Radiograph classification.

    Attributes:
        predicted_class_index: The predicted integer class label.
        predicted_class_name: The human-readable string name of the predicted class.
        confidence: The confidence score (max probability) in range [0, 1].
        class_probabilities: Mapping from class names to their prediction probabilities.
        logits: List of raw unnormalized logits for each class.
        embedding: Optional 512-D float32 GRLM embedding vector.
        processed_image: Optional preprocessed/enhanced image as a uint8 NumPy array.
    """
    predicted_class_index: int
    predicted_class_name: str
    confidence: float
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    logits: List[float] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    processed_image: Optional[np.ndarray] = None

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """Convert the prediction result to a serializable dictionary.

        Args:
            include_arrays: If True, includes NumPy arrays (embedding/processed_image) in output.

        Returns:
            Dictionary containing prediction summary.
        """
        result = {
            "predicted_class_index": self.predicted_class_index,
            "predicted_class_name": self.predicted_class_name,
            "confidence": self.confidence,
            "class_probabilities": self.class_probabilities,
            "logits": self.logits,
        }
        if include_arrays:
            if self.embedding is not None:
                result["embedding"] = self.embedding.tolist()
            if self.processed_image is not None:
                result["processed_image"] = self.processed_image.tolist()
        return result

    def __repr__(self) -> str:
        return (
            f"PredictionResult(class_index={self.predicted_class_index}, "
            f"class_name='{self.predicted_class_name}', "
            f"confidence={self.confidence:.4f})"
        )
