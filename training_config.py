from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TrainingConfig:
    img_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    scheduler: str  # none | cosine | step
    step_size: int
    step_gamma: float
    early_stop_patience: int
    strong_augmentation: bool
    hog_orientations: int = 9
    hog_pixels_per_cell: int = 8
    hog_cells_per_block: int = 2


BASE = TrainingConfig(
    img_size=224,
    epochs=30,
    batch_size=32,
    learning_rate=5e-4,
    weight_decay=5e-4,
    scheduler="cosine",
    step_size=5,
    step_gamma=0.5,
    early_stop_patience=7,
    strong_augmentation=True,
)

PER_MODEL: Dict[str, TrainingConfig] = {
    # Solid generalist; moderate lr and augmentation
    "resnet18": TrainingConfig(
        img_size=224,
        epochs=40,
        batch_size=32,
        learning_rate=5e-4,
        weight_decay=5e-4,
        scheduler="cosine",
        step_size=5,
        step_gamma=0.5,
        early_stop_patience=7,
        strong_augmentation=True,
    ),
    # Lightweight; slightly higher lr to converge faster
    "mobilenet_v3_small": TrainingConfig(
        img_size=224,
        epochs=35,
        batch_size=40,
        learning_rate=7e-4,
        weight_decay=5e-4,
        scheduler="cosine",
        step_size=4,
        step_gamma=0.6,
        early_stop_patience=6,
        strong_augmentation=True,
    ),
    # Strongest backbone; conservative lr, strong aug
    "efficientnet_b0": TrainingConfig(
        img_size=224,
        epochs=45,
        batch_size=32,
        learning_rate=3e-4,
        weight_decay=5e-4,
        scheduler="cosine",
        step_size=5,
        step_gamma=0.5,
        early_stop_patience=7,
        strong_augmentation=True,
    ),
}


def get_config(model_name: str) -> TrainingConfig:
    """Return per-model config with sensible defaults."""
    return PER_MODEL.get(model_name, BASE)
