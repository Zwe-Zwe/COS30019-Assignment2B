from dataclasses import dataclass


@dataclass(frozen=True)
class SharedTrainingConfig:
    """
    Centralised configuration so every model trains under the same regime.
    Adjust values here to run new experiments consistently across scripts.
    """
    img_size: int = 224
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 5e-4
    scheduler: str = "cosine"  # choices: none | cosine | step
    step_size: int = 5
    step_gamma: float = 0.5
    early_stop_patience: int = 7
    strong_augmentation: bool = True
    feature_batch_size: int = 48  # for feature extractors / classical models
    hog_orientations: int = 9
    hog_pixels_per_cell: int = 8
    hog_cells_per_block: int = 2


CONFIG = SharedTrainingConfig()
