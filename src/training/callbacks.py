import logging
from typing import Dict, List, Optional

import wandb
from gliner.training import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class EntityF1Callback(TrainerCallback):
    """Custom GLiNER training callback for per-entity F1 logging, early stopping, and checkpointing."""

    def __init__(
        self,
        entity_types: List[str],
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.001,
        checkpoint_metadata: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            entity_types: List of entity type labels to track individually.
            early_stopping_patience: Epochs without improvement before stopping.
            early_stopping_threshold: Minimum improvement in macro-F1 to reset patience.
            checkpoint_metadata: Extra metadata dict to persist alongside each checkpoint.
        """
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
        Compute and log per-entity-type F1 scores to W&B at the end of each epoch.

        Args:
            args: HuggingFace / GLiNER TrainingArguments.
            state: Current TrainerState (contains log history and epoch).
            control: TrainerControl used to signal early stopping.

        Returns:
            Possibly mutated TrainerControl with should_training_stop set.
        """
        # TODO: Extract per-entity-type predictions and references from kwargs or state.
        # TODO: Compute F1 for each entity type in self.entity_types.
        # TODO: Log a dict of {f"f1/{etype}": score, ...} to wandb.log.
        # TODO: Compute macro-average F1 and log as "f1/macro".
        # TODO: Delegate to _check_early_stopping and return updated control.
        raise NotImplementedError

    def _check_early_stopping(
        self, macro_f1: float, control: TrainerControl
    ) -> TrainerControl:
        """
        Update patience counter and set control.should_training_stop if needed.

        Args:
            macro_f1: Current epoch macro-average F1.
            control: TrainerControl to mutate.

        Returns:
            Updated TrainerControl.
        """
        # TODO: If macro_f1 > self._best_f1 + self.early_stopping_threshold, reset counter.
        # TODO: Otherwise increment self._epochs_without_improvement.
        # TODO: If counter >= self.early_stopping_patience, set control.should_training_stop = True.
        raise NotImplementedError

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """
        Write a metadata JSON file alongside each saved checkpoint.

        Args:
            args: TrainingArguments containing output_dir.
            state: TrainerState with current epoch and best metric.
            control: TrainerControl (not mutated).
        """
        # TODO: Build metadata dict from self.checkpoint_metadata + epoch, best_f1, global_step.
        # TODO: Determine the latest checkpoint directory under args.output_dir.
        # TODO: Write metadata as checkpoint_dir/training_metadata.json.
        raise NotImplementedError
