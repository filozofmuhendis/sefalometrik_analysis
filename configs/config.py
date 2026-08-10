"""Global configuration module for Cephalometric Deep Learning Pipeline.

All hyperparameters, operational constants, dataset settings, and ablation study controls
are defined here to eliminate magic numbers and ensure strict adherence to clean architecture.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class DatasetConfig:
    """Hyperparameters and paths for Real Cephalometric Dataset Integration.

    Attributes:
        dataset_root: Path to root dataset directory.
        train_manifest: Path to train split CSV manifest.
        val_manifest: Path to validation split CSV manifest.
        test_manifest: Path to test split CSV manifest.
        class_mapping_file: Path to class index JSON mapping file.
        preprocessing_mode: 'online' (on-the-fly) or 'cached' (pre-rendered).
        target_size: Model input spatial resolution (Height, Width).
        imagenet_mean: RGB mean values for ImageNet normalization.
        imagenet_std: RGB standard deviation values for ImageNet normalization.
    """

    dataset_root: str = "dataset"
    train_manifest: str = "splits/train.csv"
    val_manifest: str = "splits/val.csv"
    test_manifest: str = "splits/test.csv"
    reports_dir: str = "reports"
    splits_dir: str = "splits"
    class_mapping_file: str = "reports/class_mapping.json"
    preprocessing_mode: str = "online"
    target_size: Tuple[int, int] = (512, 512)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ImageEnhancementConfig:
    """Hyperparameters for ImageEnhancement module."""

    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    bilateral_d: int = 9
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    directional_kernel_size: int = 9
    energy_power_exponent: float = 1.5
    percentile_limit: float = 99.5
    fusion_l_weight: float = 0.65
    fusion_edge_weight: float = 0.35
    gaussian_blur_kernel_size: Tuple[int, int] = (3, 3)
    gaussian_blur_sigma: float = 0.8
    epsilon: float = 1e-8


@dataclass
class BackboneConfig:
    """Hyperparameters for EfficientNetV2 Backbone module."""

    model_name: str = "efficientnetv2_s"
    pretrained: bool = True
    out_channels: int = 256
    progressive_phase: int = 1


@dataclass
class AFEMConfig:
    """Hyperparameters for Adaptive Feature Enhancement Module (AFEM)."""

    in_channels: int = 256
    reduction_ratio: int = 16
    low_kernel_size: int = 3
    high_dw_kernel_size: int = 5
    epsilon: float = 1e-8


@dataclass
class CFFMConfig:
    """Hyperparameters for Cross Scale Feature Fusion Module (CFFM)."""

    in_channels: int = 256
    target_level: str = "P2"
    num_fuzzy_sets: int = 3
    fuzzy_hidden_dim: int = 32
    epsilon: float = 1e-8


@dataclass
class ACAMConfig:
    """Hyperparameters for Adaptive Context Aggregation Module (ACAM)."""

    in_channels: int = 256
    local_kernel_size: int = 3
    epsilon: float = 1e-8


@dataclass
class GRLMConfig:
    """Hyperparameters for Global Representation Learning Module (GRLM)."""

    in_channels: int = 256
    embedding_dim: int = 512
    dropout_rate: float = 0.2
    epsilon: float = 1e-8


@dataclass
class ClassificationHeadConfig:
    """Hyperparameters for Classification Head module."""

    in_features: int = 512
    num_classes: int = 3
    dropout_rate: float = 0.3


@dataclass
class AblationConfig:
    """Ablation Study Configuration Controls."""

    use_afem: bool = True
    use_cffm: bool = True
    use_acam: bool = True
    use_grlm: bool = True
    cffm_fusion_method: str = "fuzzy"
    afem_mode: str = "full"
    acam_mode: str = "full"
    global_seed: int = 42


@dataclass
class TrainingConfig:
    """Hyperparameters for model training pipeline."""

    num_epochs: int = 30
    phase1_epochs: int = 10
    phase2_epochs: int = 20
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    eta_min: float = 1e-6
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 7
    early_stopping_delta: float = 1e-4
    use_mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs/cephalometric_experiment"


@dataclass
class Config:
    """Master configuration class aggregating all pipeline sub-configurations."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    image_enhancement: ImageEnhancementConfig = field(default_factory=ImageEnhancementConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    afem: AFEMConfig = field(default_factory=AFEMConfig)
    cffm: CFFMConfig = field(default_factory=CFFMConfig)
    acam: ACAMConfig = field(default_factory=ACAMConfig)
    grlm: GRLMConfig = field(default_factory=GRLMConfig)
    head: ClassificationHeadConfig = field(default_factory=ClassificationHeadConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
