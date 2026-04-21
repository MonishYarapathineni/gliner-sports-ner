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
    """

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        others_lr=config.others_lr,
        weight_decay=config.weight_decay,
        others_weight_decay=config.others_weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        eval_strategy="epoch",         
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        fp16=config.fp16,
        seed=config.seed,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        report_to="wandb",
    )

    callback = EntityF1Callback(
        entity_types=config.entity_types,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_threshold=config.early_stopping_threshold,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=[callback],
    )

    return trainer

def convert_to_gliner_format(examples: List[Dict]) -> List[Dict]:
    """
    Convert our annotated format to GLiNER's native training format.
    
    Our format:
        {"text": str, "entities": [{"start": int, "end": int, "label": str}]}
    
    GLiNER format:
        {"tokenized_text": List[str], "ner": [[start_tok, end_tok, label]]}
    
    Args:
        examples: List of our annotated examples.
    
    Returns:
        List of GLiNER-format dicts.
    """
    converted = []

    for example in examples:
        text = example.get("text", "")
        entities = example.get("entities", [])

        # Whitespace tokenize — must match how GLiNER tokenizes internally
        tokens = text.split()

        # Build char-to-token index map
        char_to_token = {}
        char_pos = 0
        for i, token in enumerate(tokens):
            for _ in token:
                char_to_token[char_pos] = i
                char_pos += 1
            char_pos += 1  # space

        ner = []
        for ent in entities:
            start_char = ent.get("start", -1)
            end_char = ent.get("end", -1)
            label = ent.get("label", "")

            # Map char offsets to token indices
            start_tok = char_to_token.get(start_char)

            # end_char is exclusive, so look up end_char - 1
            end_tok = char_to_token.get(end_char - 1)

            if start_tok is None or end_tok is None:
                continue

            ner.append([start_tok, end_tok, label])

        if not ner:
            continue

        converted.append({
            "tokenized_text": tokens,
            "ner": ner,
        })

    return converted

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
        config=vars(config),
    )

    logger.info("Loading datasets...")
    train_raw = load_data(config.train_path)
    val_raw = load_data(config.val_path)

    # Convert to GLiNER native format
    train_data = convert_to_gliner_format(train_raw)
    val_data = convert_to_gliner_format(val_raw)

    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")

    model = initialize_model(config)
    trainer = build_trainer(model, train_data, val_data, config)

    logger.info("Starting training...")
    trainer.train()

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