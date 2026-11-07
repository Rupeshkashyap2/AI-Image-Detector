"""
src/training/config.py

All training hyperparameters and project paths in one place.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "dataset" / "processed"
SAVED_MODELS_DIR = ROOT / "saved_models"
GRAPHS_DIR = ROOT / "results" / "graphs"


@dataclass
class TrainConfig:
    # Model
    arch: str = "efficientnet_b0"
    num_classes: int = 2
    pretrained: bool = True
    dropout: float = 0.2

    # Data
    batch_size: int = 32          # lower to 16 if you hit a CUDA out-of-memory error
    num_workers: int = 4          # use 0 if the DataLoader hangs on Windows

    # Stage 1: train the classification head, backbone frozen
    head_epochs: int = 3
    head_lr: float = 1e-3

    # Stage 2: fine-tune the last backbone blocks
    finetune_epochs: int = 17
    finetune_lr: float = 1e-4
    unfreeze_blocks: int = 4      # last N blocks of model.features (the net has 9)

    # Optimization
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    early_stop_patience: int = 5  # counted in fine-tune epochs only

    # Misc
    use_amp: bool = True
    seed: int = 42
    max_batches: int = 0          # 0 = full epochs; >0 limits batches (smoke test)
    run_name: str = "efficientnet"

    def as_dict(self) -> dict:
        return asdict(self)