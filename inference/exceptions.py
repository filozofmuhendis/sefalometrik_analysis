"""Custom Exceptions for Cephalometric Radiograph Inference Pipeline.

Defines a hierarchy of domain-specific exceptions to handle loading, preprocessing,
configuration, validation, and inference failures gracefully.
"""

class CephalometricInferenceError(Exception):
    """Base exception for all cephalometric radiograph inference pipeline errors."""
    pass


class ImageLoadError(CephalometricInferenceError):
    """Raised when an input image cannot be loaded from the specified path or source."""
    pass


class InvalidImageError(CephalometricInferenceError):
    """Raised when an input image fails validation (e.g., empty, wrong type, NaN/Inf, or invalid shape)."""
    pass


class ModelLoadError(CephalometricInferenceError):
    """Raised when the model weights or checkpoint fails to load correctly."""
    pass


class ConfigurationError(CephalometricInferenceError):
    """Raised when there is a mismatch between checkpoint configuration and runtime configuration."""
    pass


class PreprocessingError(CephalometricInferenceError):
    """Raised when an error occurs during image preprocessing or enhancement."""
    pass


class InferenceError(CephalometricInferenceError):
    """Raised when an unexpected error occurs during model forward execution."""
    pass
