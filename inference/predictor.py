"""Cephalometric Analysis Inference Predictor Engine.

Loads trained model checkpoints and executes end-to-end inference on single or batch
cephalometric lateral X-ray radiograph images. Implements input validation, image enhancement,
tensor preprocessing, numerical safety checks, and returns structured PredictionResult objects.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from configs.config import Config
from models.cephalometric_net import CephalometricAnalysisNet
from modules.image_enhancement import ImageEnhancement
from inference.exceptions import (
    ImageLoadError,
    InvalidImageError,
    ModelLoadError,
    ConfigurationError,
    PreprocessingError,
    InferenceError,
)
from inference.result import PredictionResult

# Initialize local logger
logger = logging.getLogger("CephalometricPredictor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class CephalometricPredictor:
    """Production-ready Inference Engine for Cephalometric X-ray Classification."""

    def __init__(
        self,
        config: Optional[Config] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "auto",
        confidence_threshold: Optional[float] = None,
    ) -> None:
        """Initialize the predictor, load the model architecture and restore checkpoint weights.

        Args:
            config: Master Config instance. If None, default Config is instantiated.
            checkpoint_path: Optional path to checkpoint (.pth). If None, reads from config or defaults.
            device: Target device. 'auto' selects CUDA if available, otherwise CPU.
            confidence_threshold: Optional threshold for confidence warnings/status.

        Raises:
            ModelLoadError: If loading model weights or checkpoint fails.
            ConfigurationError: If there is a configuration/model structure mismatch.
        """
        self.config = config if config is not None else Config()
        self.confidence_threshold = confidence_threshold

        # Setup device
        if device == "auto" or isinstance(device, str) and device.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize Image Enhancement Pipeline
        self.enhancer = ImageEnhancement(config=self.config.image_enhancement)

        # Initialize Model Architecture
        try:
            self.model = CephalometricAnalysisNet(config=self.config)
        except Exception as e:
            raise ConfigurationError(f"Failed to instantiate model architecture: {str(e)}") from e

        # Class Mapping / Name resolution
        self.class_mapping = self._load_class_mapping()
        self.class_names = [k for k, v in sorted(self.class_mapping.items(), key=lambda item: item[1])]

        # Load weights
        checkpoint_to_load = checkpoint_path
        if checkpoint_to_load is None:
            # Try to get from config dataset or training checkpoint_dir
            checkpoint_to_load = os.environ.get("MODEL_CHECKPOINT", None)
            if checkpoint_to_load is None:
                # Check for standard best checkpoint paths
                default_paths = [
                    Path("checkpoints/baseline0/best_checkpoint.pth"),
                    Path("checkpoints/best_model.pth"),
                ]
                for p in default_paths:
                    if p.exists():
                        checkpoint_to_load = p
                        break

        if checkpoint_to_load is not None:
            self._load_checkpoint(checkpoint_to_load)
        else:
            logger.warning("No checkpoint path provided or discovered. Running with randomly initialized weights!")

        self.model.to(self.device)
        self.model.eval()

    def _load_class_mapping(self) -> Dict[str, int]:
        """Resolve class mapping from report directory or fallback to defaults."""
        mapping_path = Path(self.config.dataset.class_mapping_file)
        if mapping_path.exists():
            try:
                import json
                with open(mapping_path, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                return mapping
            except Exception as e:
                logger.warning(f"Failed to load class mapping from {mapping_path}: {e}. Using defaults.")
        
        # Default fallback mapping
        return {"sinif1": 0, "sinif2": 1, "sinif3": 2}

    def _load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load and validate model weights from checkpoint.

        Args:
            path: Path to checkpoint file.

        Raises:
            ModelLoadError: If loading weights fails.
            ConfigurationError: If checkpoint structure is incompatible.
        """
        path = Path(path)
        if not path.exists():
            raise ModelLoadError(f"Checkpoint file does not exist: {path}")

        try:
            # Map location keeps weights clean between devices
            checkpoint_data = torch.load(path, map_location=self.device)
        except Exception as e:
            raise ModelLoadError(f"Failed to read checkpoint file: {str(e)}") from e

        # Extract state dict
        if "model_state_dict" in checkpoint_data:
            state_dict = checkpoint_data["model_state_dict"]
        else:
            state_dict = checkpoint_data

        # Inspect checkpoint config for consistency checks
        if "config" in checkpoint_data and checkpoint_data["config"] is not None:
            ckpt_config = checkpoint_data["config"]
            
            # Check class counts
            if "class_mapping" in checkpoint_data and checkpoint_data["class_mapping"] is not None:
                ckpt_mapping = checkpoint_data["class_mapping"]
                if len(ckpt_mapping) != len(self.class_names):
                    raise ConfigurationError(
                        f"Class count mismatch: checkpoint has {len(ckpt_mapping)} classes, "
                        f"but runtime config specifies {len(self.class_names)} classes."
                    )
            
            # Check ablation mismatch
            ckpt_ablation = None
            if isinstance(ckpt_config, dict):
                ckpt_ablation = ckpt_config.get("ablation", None)
            elif hasattr(ckpt_config, "ablation"):
                ckpt_ablation = ckpt_config.ablation

            if ckpt_ablation is not None:
                if isinstance(ckpt_ablation, dict):
                    ckpt_afem = ckpt_ablation.get("use_afem", True)
                    ckpt_cffm = ckpt_ablation.get("use_cffm", True)
                    ckpt_acam = ckpt_ablation.get("use_acam", True)
                    ckpt_grlm = ckpt_ablation.get("use_grlm", True)
                else:
                    ckpt_afem = getattr(ckpt_ablation, "use_afem", True)
                    ckpt_cffm = getattr(ckpt_ablation, "use_cffm", True)
                    ckpt_acam = getattr(ckpt_ablation, "use_acam", True)
                    ckpt_grlm = getattr(ckpt_ablation, "use_grlm", True)

                if (ckpt_afem != self.config.ablation.use_afem or
                    ckpt_cffm != self.config.ablation.use_cffm or
                    ckpt_acam != self.config.ablation.use_acam or
                    ckpt_grlm != self.config.ablation.use_grlm):
                    raise ConfigurationError(
                        f"Ablation setting mismatch: Checkpoint [AFEM={ckpt_afem}, CFFM={ckpt_cffm}, "
                        f"ACAM={ckpt_acam}, GRLM={ckpt_grlm}] vs Runtime config [AFEM={self.config.ablation.use_afem}, "
                        f"CFFM={self.config.ablation.use_cffm}, ACAM={self.config.ablation.use_acam}, "
                        f"GRLM={self.config.ablation.use_grlm}]."
                    )

        # Load weights into model
        try:
            self.model.load_state_dict(state_dict)
            logger.info(f"Successfully loaded model checkpoint from: {path} (device: {self.device})")
        except Exception as e:
            raise ModelLoadError(f"State dict loading failed: {str(e)}") from e

        # If class mapping is saved in checkpoint, override mapping to ensure compatibility
        if "class_mapping" in checkpoint_data and checkpoint_data["class_mapping"] is not None:
            self.class_mapping = checkpoint_data["class_mapping"]
            self.class_names = [k for k, v in sorted(self.class_mapping.items(), key=lambda item: item[1])]

    def _validate_image(self, image: Any) -> np.ndarray:
        """Validate input image type, shape, and numerical values.

        Returns:
            np.ndarray: Verified image in BGR format.

        Raises:
            InvalidImageError: If the image fails structure/content checks.
            ImageLoadError: If loading image path fails.
        """
        if image is None:
            raise InvalidImageError("Input image cannot be None.")

        # Check if PIL Image
        try:
            from PIL import Image
            if isinstance(image, Image.Image):
                img_np = np.array(image)
                if img_np.ndim == 2:
                    return cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                elif img_np.ndim == 3 and img_np.shape[2] == 3:
                    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                elif img_np.ndim == 3 and img_np.shape[2] == 4:
                    return cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                else:
                    raise InvalidImageError(f"Unsupported PIL Image shape: {img_np.shape}")
        except ImportError:
            pass

        # Check if file path
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise ImageLoadError(f"Image file path does not exist: {path}")
            if not path.is_file():
                raise ImageLoadError(f"Image path is not a file: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise ImageLoadError(f"Failed to load or decode image file from path: {path}")
            return img

        # Check if NumPy array
        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise InvalidImageError("Input image NumPy array is empty.")
            
            # Check dimensions
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3:
                if image.shape[2] == 1:
                    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 3:
                    return image.copy()
                elif image.shape[2] == 4:
                    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                else:
                    raise InvalidImageError(f"Invalid number of channels in image array: {image.shape[2]}")
            else:
                raise InvalidImageError(f"Invalid image array dimension: {image.ndim} (expected 2 or 3)")

        raise InvalidImageError(f"Unsupported image type: {type(image)}")

    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Apply mathematical image enhancement, resize, and normalization.

        Returns:
            Tuple of:
                torch.Tensor: Preprocessed input tensor shape [3, H, W]
                np.ndarray: Enhanced grayscale image array shape [H, W]
        """
        # Step 1: Image Enhancement Pipeline
        try:
            enhanced_gray = self.enhancer.process(img_bgr)
        except Exception as e:
            raise PreprocessingError(f"Image enhancement pipeline failed: {str(e)}") from e

        # Ensure 3 channels as expected by model (Gray to BGR)
        enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        # Step 2: Resize
        target_h, target_w = self.config.dataset.target_size
        resized = cv2.resize(enhanced_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Step 3: [0, 1] Scaling & Normalization
        img_float = resized.astype(np.float32) / 255.0

        mean = np.array(self.config.dataset.imagenet_mean, dtype=np.float32)
        std = np.array(self.config.dataset.imagenet_std, dtype=np.float32)
        normalized = (img_float - mean) / std

        # Step 4: Transpose to [C, H, W]
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
        return tensor, enhanced_gray

    def predict(
        self,
        image: Any,
        return_embedding: bool = False,
        return_processed_image: bool = False,
        top_k: Optional[int] = None,
    ) -> PredictionResult:
        """Execute inference on a single input image.

        Args:
            image: Image path (str/Path), PIL Image, or BGR NumPy array.
            return_embedding: If True, returns GRLM 512-D embedding in PredictionResult.
            return_processed_image: If True, returns enhanced uint8 image in PredictionResult.
            top_k: If specified, returns top-K probabilities.

        Returns:
            PredictionResult: Structured dataclass representing prediction results.

        Raises:
            CephalometricInferenceError: Base class for all handled failures.
        """
        # Validate input image
        img_bgr = self._validate_image(image)

        # Preprocess
        tensor_3ch, enhanced_gray = self._preprocess(img_bgr)

        # Add batch dimension and transfer to device
        input_batch = tensor_3ch.unsqueeze(0).to(self.device)

        # Run model forward pass under inference mode (no gradients)
        try:
            with torch.no_grad():
                embedding_tensor = self.model.extract_embedding(input_batch)
                logits_tensor = self.model.head(embedding_tensor)
                
                # Check for numerical stability in raw outputs
                if torch.isnan(logits_tensor).any() or torch.isinf(logits_tensor).any():
                    raise InferenceError("Model generated NaN or Inf logits values during forward pass.")

                probs_tensor = F.softmax(logits_tensor, dim=1)

                if torch.isnan(probs_tensor).any() or torch.isinf(probs_tensor).any():
                    raise InferenceError("Probability computation produced NaN or Inf values.")
        except CephalometricInferenceError:
            raise
        except Exception as e:
            raise InferenceError(f"Model forward pass failed: {str(e)}") from e

        # Extract values to CPU
        logits = logits_tensor.squeeze(0).cpu().numpy().tolist()
        probs_np = probs_tensor.squeeze(0).cpu().numpy()
        embedding_np = embedding_tensor.squeeze(0).cpu().numpy()

        # Validate probability bounds
        if not np.all(np.isfinite(probs_np)):
            raise InferenceError("Probability array contains non-finite values.")
        if np.any(probs_np < 0.0):
            raise InferenceError("Probability array contains negative values.")
        if not np.isclose(np.sum(probs_np), 1.0, atol=1e-4):
            raise InferenceError(f"Probability vector sum deviates from 1.0 (sum={np.sum(probs_np):.4f}).")

        # Select predicted class
        pred_idx = int(np.argmax(probs_np))
        confidence = float(probs_np[pred_idx])
        pred_class_name = self.class_names[pred_idx]

        # Log prediction result in privacy-safe format
        logger.info(
            f"Prediction complete. Predicted class: '{pred_class_name}' "
            f"(idx: {pred_idx}) | Prediction Confidence: {confidence*100:.2f}% | "
            f"Device: {self.device}"
        )

        # Map class names to probabilities
        class_probabilities = {self.class_names[i]: float(probs_np[i]) for i in range(len(self.class_names))}

        # If confidence threshold is exceeded, warn
        if self.confidence_threshold is not None and confidence < self.confidence_threshold:
            logger.warning(
                f"Prediction confidence ({confidence*100:.2f}%) is below the configured threshold "
                f"({self.confidence_threshold*100:.2f}%)."
            )

        # Prepare outputs
        embedding = embedding_np if return_embedding else None
        processed_image = enhanced_gray if return_processed_image else None

        return PredictionResult(
            predicted_class_index=pred_idx,
            predicted_class_name=pred_class_name,
            confidence=confidence,
            class_probabilities=class_probabilities,
            logits=logits,
            embedding=embedding,
            processed_image=processed_image,
        )

    def predict_batch(
        self,
        images: List[Any],
        return_embeddings: bool = False,
        return_processed_images: bool = False,
    ) -> List[PredictionResult]:
        """Execute inference on a list of images using the same model instance.

        Args:
            images: List of image paths (str/Path), PIL Images, or BGR NumPy arrays.
            return_embeddings: If True, returns embeddings in each PredictionResult.
            return_processed_images: If True, returns enhanced images.

        Returns:
            List[PredictionResult]: Structured results for each input image.
        """
        results = []
        for i, img in enumerate(images):
            try:
                res = self.predict(
                    image=img,
                    return_embedding=return_embeddings,
                    return_processed_image=return_processed_images,
                )
                results.append(res)
            except Exception as e:
                logger.error(f"Batch index {i} failed prediction: {str(e)}")
                # Append a dummy or None to preserve mapping index
                results.append(None)
        return results
