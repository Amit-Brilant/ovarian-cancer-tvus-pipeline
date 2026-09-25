"""
Configuration management for ovarian carcinoma detection system.
Uses dataclasses for type-safe, modular configuration.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def resolve_seed(seed: Optional[int] = None) -> int:
    """Seeds are not stored in the code: pass --seed or set the SEED environment variable."""
    if seed is None:
        seed = os.environ.get("SEED")
    if seed is None:
        raise SystemExit("no random seed given: pass --seed or set the SEED environment variable")
    return int(seed)


@dataclass
class DataConfig:
    """Data-related configuration."""

    # Data paths
    data_root: str = r"/path/to/project/Thesis code"

    # Image preprocessing
    image_size: Tuple[int, int] = (224, 224)  # Resize from 1240x1240 to match ImageNet pretrained models

    # Data splitting ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # DataLoader settings
    batch_size: int = 32
    num_workers: int = 1
    shuffle: bool = True
    seed: Optional[int] = None

    # Augmentation flags - all toggleable for ablation studies
    use_augmentation: bool = False
    aug_horizontal_flip: bool = False
    aug_vertical_flip: bool = False
    aug_rotation: bool = False
    aug_rotation_degrees: int = 0
    aug_color_jitter: bool = False
    aug_color_jitter_brightness: float = 0.2
    aug_color_jitter_contrast: float = 0.2
    aug_color_jitter_saturation: float = 0.2
    aug_color_jitter_hue: float = 0.1
    aug_gaussian_blur: bool = False
    aug_gaussian_blur_kernel: int = 3
    aug_random_erasing: bool = False
    aug_random_erasing_prob: float = 0.5
    out_split_dir: str = "./split_artifacts"



@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Architecture selection
    architecture: str = "resnet50"  # Options: resnet18/34/50/101/152, efficientnet_b0-b7
    pretrained: bool = True  # Use ImageNet pretrained weights
    num_classes: int = 1  # Binary classification (single sigmoid output)
    dropout_rate: float = 0.5

    # BatchNorm settings (for experimentation)
    bn_track_running_stats: bool = True
    bn_momentum: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Training duration
    num_epochs: int = 100

    # Optimization
    learning_rate: float = 1e-4
    optimizer: str = "adam"  # Options: adam, sgd, adamw
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD

    # Loss function
    loss_function: str = "bce_with_logits"  # Options: bce_with_logits, focal
    focal_alpha: float = 0.25  # For focal loss
    focal_gamma: float = 2.0   # For focal loss

    # Learning rate scheduler
    scheduler: str = "reduce_on_plateau"  # Options: step, reduce_on_plateau, cosine, none
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_step_size: int = 10
    scheduler_min_lr: float = 1e-6

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    early_stopping_mode: str = 'max'  # 'max' for F1, 'min' for loss


@dataclass
class ExperimentConfig:
    """Experiment tracking and logging configuration."""

    # Experiment identification
    experiment_name: str = "stage1_binary_classification"

    # Weights & Biases settings
    use_wandb: bool = True
    wandb_project: str = "ovarian-carcinoma-detection"
    wandb_entity: Optional[str] = None  # Your wandb username (optional)

    # Checkpoint management
    save_checkpoints: bool = True
    checkpoint_dir: str = r"./experiments/checkpoints"
    save_best_only: bool = True

    # Evaluation and visualization
    compute_gradcam: bool = True
    gradcam_num_samples: int = 10
    gradcam_layer: Optional[str] = None  # Auto-detect based on architecture if None


@dataclass
class GridSearchConfig:
    """Grid search hyperparameter tuning configuration."""

    enable_grid_search: bool = False

    # Hyperparameter grids (lists of values to try)
    learning_rates: List[float] = field(default_factory=lambda: [1e-3, 1e-4, 1e-5])
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64])
    architectures: List[str] = field(default_factory=lambda: ["resnet50", "efficientnet_b0"])
    optimizers: List[str] = field(default_factory=lambda: ["adam", "adamw"])
    dropout_rates: List[float] = field(default_factory=lambda: [0.3, 0.5])

    # Grid search constraints
    max_trials: int = 50  # Maximum combinations to try (for computational limits)
    metric_to_optimize: str = "val_visit_f1"  # Options: val_loss, val_visit_f1, val_visit_auc


@dataclass
class KFoldConfig:
    """K-Fold cross-validation configuration."""

    enable_kfold: bool = False
    n_folds: int = 5
    fold_to_run: Optional[int] = None  # If None, run all folds; otherwise run specific fold
