import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import wandb
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class EntityF1Callback(TrainerCallback):
    """Custom GLiNER training callback for per-entity F1 logging and early stopping."""

    def __init__(
        self,
        entity_types: List[str],
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.001,
        checkpoint_metadata: Optional[Dict] = None,
    ) -> None:
        self.entity_types = entity_types
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.checkpoint_metadata = checkpoint_metadata or {}
        self._best_f1: float = 0.0
        self._epochs_without_improvement: int = 0

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """
        Log per-entity F1 and macro F1 to W&B. Trigger early stopping if needed.
        """
        # Pull eval metrics from the last log entry in state
        # GLiNER trainer logs eval metrics into state.log_history each epoch
        metrics = {}
        for entry in reversed(state.log_history):
            if "eval_f1" in entry or any("eval_" in k for k in entry):
                metrics = entry
                break

        # Extract per-entity F1 if available, else fall back to overall eval_f1
        wandb_log = {}
        per_entity_f1s = []

        for entity_type in self.entity_types:
            # GLiNER logs per-entity metrics as eval_{entity_type}_f1
            key = f"eval_{entity_type.lower()}_f1"
            score = metrics.get(key, None)
            if score is not None:
                wandb_log[f"f1/{entity_type}"] = score
                per_entity_f1s.append(score)

        # Macro F1 — average across entity types if available, else use overall
        if per_entity_f1s:
            macro_f1 = sum(per_entity_f1s) / len(per_entity_f1s)
        else:
            macro_f1 = metrics.get("eval_f1", 0.0)

        wandb_log["f1/macro"] = macro_f1
        wandb_log["epoch"] = state.epoch

        if wandb.run is not None:
            wandb.log(wandb_log)

        logger.info(f"Epoch {state.epoch:.0f} — macro F1: {macro_f1:.4f}")

        return self._check_early_stopping(macro_f1, control)

    def _check_early_stopping(
        self, macro_f1: float, control: TrainerControl
    ) -> TrainerControl:
        """
        Update patience and stop training if no improvement.
        """
        if macro_f1 > self._best_f1 + self.early_stopping_threshold:
            self._best_f1 = macro_f1
            self._epochs_without_improvement = 0
            logger.info(f"New best macro F1: {macro_f1:.4f} — patience reset")
        else:
            self._epochs_without_improvement += 1
            logger.info(
                f"No improvement. Patience: "
                f"{self._epochs_without_improvement}/{self.early_stopping_patience}"
            )

        if self._epochs_without_improvement >= self.early_stopping_patience:
            logger.info("Early stopping triggered")
            control.should_training_stop = True

        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """
        Write metadata JSON alongside each saved checkpoint.
        """
        metadata = {
            **self.checkpoint_metadata,
            "epoch": state.epoch,
            "best_f1": self._best_f1,
            "global_step": state.global_step,
        }

        # Find latest checkpoint directory
        output_path = Path(args.output_dir)
        checkpoints = sorted(output_path.glob("checkpoint-*"))

        if checkpoints:
            checkpoint_dir = checkpoints[-1]
            metadata_path = checkpoint_dir / "training_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")