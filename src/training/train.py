import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

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
        path: Path to a JSONL file where each line is a training example dict.

    Returns:
        List of GLiNER training example dicts.
    """
    # TODO: Open the file and parse each line with json.loads.
    # TODO: Return the list of dicts.
    raise NotImplementedError


def initialize_model(config: TrainingConfig) -> GLiNER:
    """
    Load the GLiNER model from a pretrained checkpoint.

    Args:
        config: TrainingConfig with model_name and cache_dir.

    Returns:
        Loaded GLiNER model instance.
    """
    # TODO: Call GLiNER.from_pretrained(config.model_name, cache_dir=config.cache_dir).
    # TODO: Return the model.
    raise NotImplementedError


def build_trainer(
    model: GLiNER,
    train_data: List[Dict],
    val_data: List[Dict],
    config: TrainingConfig,
) -> Trainer:
    """
    Construct a GLiNER Trainer with the provided data and config.

    Args:
        model: Initialized GLiNER model.
        train_data: Training examples.
        val_data: Validation examples.
        config: TrainingConfig with all hyperparameters.

    Returns:
        Configured GLiNER Trainer instance.
    """
    # TODO: Build TrainingArguments from config fields
    #       (output_dir, num_train_epochs, learning_rate, per_device_train_batch_size, etc.).
    # TODO: Instantiate EntityF1Callback with config.entity_types.
    # TODO: Construct and return Trainer(model, args, train_data, val_data, callbacks=[...]).
    raise NotImplementedError


def run_experiment(config: TrainingConfig) -> None:
    """
    Initialize W&B, load data, build trainer, and train the model.

    Args:
        config: Fully populated TrainingConfig.
    """
    # TODO: Call wandb.init with project=config.wandb_project, name=config.wandb_run_name,
    #       and config=vars(config).
    # TODO: Load train and val data with load_data.
    # TODO: Initialize model with initialize_model.
    # TODO: Build trainer with build_trainer.
    # TODO: Call trainer.train().
    # TODO: Save final model to config.output_dir.
    # TODO: Call wandb.finish().
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments that override TrainingConfig defaults.

    Returns:
        Parsed argparse.Namespace.
    """
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
    """
    Entry point: build config from defaults + CLI overrides, then run experiment.
    """
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = TrainingConfig()

    # TODO: Apply non-None CLI overrides onto config fields.
    # TODO: Call run_experiment(config).
    raise NotImplementedError


if __name__ == "__main__":
    main()
