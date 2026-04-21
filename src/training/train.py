import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import wandb
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments

from src.training.config import TrainingConfig
from src.training.callbacks import EntityF1Callback

logger = logging.getLogger(__name__)


def load_data(path: str) -> List[Dict]:
    """
    Load GLiNER-format examples from a JSONL file.

    Args:
        path: Path to JSONL file.

    Returns:
        List of GLiNER training example dicts.
    """
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return examples


def initialize_model(config: TrainingConfig) -> GLiNER:
    """
    Load GLiNER from pretrained checkpoint.

    Args:
        config: TrainingConfig with model_name.

    Returns:
        Loaded GLiNER model instance.
    """
    logger.info(f"Loading model: {config.model_name}")
    model = GLiNER.from_pretrained(config.model_name)
    return model


def build_trainer(
    model: GLiNER,
    train_data: List[Dict],
    val_data: List[Dict],
    config: TrainingConfig,
) -> Trainer:
    """
    Construct GLiNER Trainer with config and callbacks.

    Args:
        model: Initialized GLiNER model.
        train_data: Training examples.
        val_data: Validation examples.
        config: TrainingConfig with all hyperparameters.

    Returns:
        Configured Trainer instance.
    """
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,         # backbone LR — low to prevent forgetting
        others_lr=config.others_lr,                 # span head LR — higher for domain adaptation
        weight_decay=config.weight_decay,
        others_weight_decay=config.others_weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=config.fp16,
        seed=config.seed,
        report_to="wandb",
    )

    callback = EntityF1Callback(entity_types=config.entity_types)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=[callback],
    )

    return trainer


def run_experiment(config: TrainingConfig) -> None:
    """
    Full training run — W&B init, load data, train, save.

    Args:
        config: Fully populated TrainingConfig.
    """
    # Init W&B — logs all hyperparameters alongside metrics
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=vars(config),    # stores all hyperparams for experiment comparison
    )

    logger.info("Loading datasets...")
    train_data = load_data(config.train_path)
    val_data = load_data(config.val_path)

    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")

    model = initialize_model(config)
    trainer = build_trainer(model, train_data, val_data, config)

    logger.info("Starting training...")
    trainer.train()

    # Save final model
    output_path = Path(config.output_dir) / "best"
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    logger.info(f"Model saved to {output_path}")

    wandb.finish()


def parse_args() -> argparse.Namespace:
    """Parse CLI overrides for TrainingConfig."""
    parser = argparse.ArgumentParser(description="Fine-tune GLiNER for sports NER")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Entry point — build config, apply CLI overrides, run experiment."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = TrainingConfig()

    # Apply CLI overrides — only override if explicitly passed
    for field in vars(args):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
            logger.info(f"CLI override: {field} = {value}")

    run_experiment(config)


if __name__ == "__main__":
    main()