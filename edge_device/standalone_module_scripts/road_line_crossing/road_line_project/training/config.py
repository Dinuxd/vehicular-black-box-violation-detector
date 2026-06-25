from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "processed_dataset" / "final_dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "training_outputs"

CLASS_NAMES = {
    0: "background",
    1: "restricted_solid_line",
    2: "dashed_or_non_restricted_line",
}

NUM_CLASSES = 3
INPUT_WIDTH = 320
INPUT_HEIGHT = 192

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CATEGORY_SAMPLE_WEIGHTS = {
    "solid_only": 1.5,
    "dashed_only": 1.5,
    "both_solid_and_dashed": 2.0,
    "no_useful_lane": 0.5,
}


@dataclass
class TrainingConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    input_width: int = INPUT_WIDTH
    input_height: int = INPUT_HEIGHT
    num_classes: int = NUM_CLASSES
    batch_size: int = 16
    num_workers: int = 2
    epochs_stage1: int = 5
    epochs_stage2: int = 10
    lr_stage1: float = 1e-3
    lr_stage2: float = 1e-4
    weight_decay: float = 1e-4
    dice_loss_weight: float = 0.5
    focal_loss_weight: float = 0.0
    focal_gamma: float = 2.0
    use_class_weights: bool = True
    use_foreground_sampler: bool = True
    augment_train: bool = False
    crop_top_fraction: float = 0.0
    mask_dilation: int = 0
    selection_metric: str = "mean_foreground_iou"
    early_stopping_patience: int = 0
    pretrained_encoder: bool = True
    unfreeze_from_layer: int = 14
    preview_count: int = 12
    preview_every: int = 1
    seed: int = 42
    device: str = "auto"

    @property
    def image_size(self) -> tuple[int, int]:
        return self.input_height, self.input_width


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def output_dirs(output_root: Path) -> dict[str, Path]:
    return {
        "root": output_root,
        "models": output_root / "models",
        "plots": output_root / "plots",
        "prediction_previews": output_root / "prediction_previews",
        "reports": output_root / "reports",
    }


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    dirs = output_dirs(output_root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
