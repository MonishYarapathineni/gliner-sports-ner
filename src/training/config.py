from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TrainingConfig:
    """All hyperparameters and paths for a GLiNER fine-tuning experiment."""

    # Model
    model_name: str = "urchade/gliner_medium-v2.1"
    output_dir: str = "checkpoints/gliner-sports"
    cache_dir: str = ".cache/models"

    # Optimizer — differential LRs for backbone vs span head
    learning_rate: float = 5e-6          # backbone — low to prevent catastrophic forgetting
    others_lr: float = 1e-5              # span head — higher for domain adaptation
    weight_decay: float = 0.01
    others_weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"

    # Schedule
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2

    # Tokenization
    max_length: int = 384
    stride: int = 128

    # Hardware
    fp16: bool = False                   # set True if on A100/V100, False if getting NaN

    # GLiNER entity schema
    entity_types: List[str] = field(default_factory=lambda: [
        "PLAYER",
        "TEAM",
        "POSITION",
        "STAT",
        "INJURY",
        "TRADE_ASSET",
        "GAME_EVENT",
        "VENUE",
        "COACH",
        "AWARD",
    ])

    # Evaluation
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    # W&B
    wandb_project: str = "gliner-sports-ner"
    wandb_entity: str = ""
    wandb_run_name: str = "gliner-sports-v1"
    report_to: str = "wandb"

    # Data paths
    train_path: str = "data/splits/train.jsonl"
    val_path: str = "data/splits/val.jsonl"
    test_path: str = "data/splits/test.jsonl"

    # Split ratios
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Logging
    logging_steps: int = 50
    save_total_limit: int = 3
    seed: int = 42

    def __post_init__(self) -> None:
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)