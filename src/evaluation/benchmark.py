import logging
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from gliner import GLiNER
from openai import OpenAI

logger = logging.getLogger(__name__)


class NERBenchmark:
    """Runs inference with base GLiNER, fine-tuned GLiNER, and GPT-4o-mini on the test set."""

    GPT_ANNOTATION_PROMPT = """
Extract named entities from the following sports text.
Return JSON: {{"entities": [{{"text": str, "label": str, "start": int, "end": int}}]}}

Entity types: PLAYER, TEAM, POSITION, STAT, INJURY, TRADE_ASSET, GAME_EVENT, VENUE, COACH, AWARD

Text: {text}
"""

    def __init__(
        self,
        base_model_name: str = "urchade/gliner_medium-v2.1",
        finetuned_model_path: str = "checkpoints/gliner-sports",
        gpt_model: str = "gpt-4o-mini",
        entity_types: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> None:
        """
        Args:
            base_model_name: HuggingFace model ID for zero-shot base GLiNER.
            finetuned_model_path: Local path to the fine-tuned GLiNER checkpoint.
            gpt_model: OpenAI model ID for the GPT baseline.
            entity_types: Entity labels to extract; defaults to the 10 sports types.
            threshold: GLiNER prediction confidence threshold.
        """
        self.base_model_name = base_model_name
        self.finetuned_model_path = finetuned_model_path
        self.gpt_model = gpt_model
        self.entity_types = entity_types or [
            "PLAYER", "TEAM", "POSITION", "STAT", "INJURY",
            "TRADE_ASSET", "GAME_EVENT", "VENUE", "COACH", "AWARD",
        ]
        self.threshold = threshold
        self.base_gliner: Optional[GLiNER] = None
        self.finetuned_gliner: Optional[GLiNER] = None
        self.openai_client: Optional[OpenAI] = None

    def load_models(self) -> None:
        """
        Load all three inference systems into memory.

        Populates self.base_gliner, self.finetuned_gliner, and self.openai_client.
        """
        # TODO: Load base GLiNER with GLiNER.from_pretrained(self.base_model_name).
        # TODO: Load fine-tuned GLiNER with GLiNER.from_pretrained(self.finetuned_model_path).
        # TODO: Instantiate self.openai_client = OpenAI().
        raise NotImplementedError

    def run_base_gliner(self, examples: List[Dict]) -> Tuple[List[List[Dict]], List[float]]:
        """
        Run the base GLiNER model over the test set and collect predictions + latencies.

        Args:
            examples: GLiNER-format test examples (dicts with 'text' and 'entities' keys).

        Returns:
            Tuple of (predictions, latencies_ms) where predictions[i] is a list of
            predicted entity dicts and latencies_ms[i] is wall-clock ms for example i.
        """
        # TODO: Iterate examples, time each self.base_gliner.predict_entities call.
        # TODO: Collect predictions and latency floats.
        raise NotImplementedError

    def run_finetuned_gliner(self, examples: List[Dict]) -> Tuple[List[List[Dict]], List[float]]:
        """
        Run the fine-tuned GLiNER model over the test set.

        Args:
            examples: GLiNER-format test examples.

        Returns:
            Tuple of (predictions, latencies_ms).
        """
        # TODO: Same structure as run_base_gliner but using self.finetuned_gliner.
        raise NotImplementedError

    def run_gpt(self, examples: List[Dict]) -> Tuple[List[List[Dict]], List[float], float]:
        """
        Run GPT-4o-mini over the test set via the OpenAI chat completions API.

        Args:
            examples: GLiNER-format test examples.

        Returns:
            Tuple of (predictions, latencies_ms, total_cost_usd).
        """
        # TODO: For each example, format GPT_ANNOTATION_PROMPT and call completions API.
        # TODO: Parse JSON response into entity list.
        # TODO: Track token usage to compute total_cost_usd using GPT-4o-mini pricing.
        raise NotImplementedError

    def estimate_gpt_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate USD cost for a GPT-4o-mini API call.

        Args:
            prompt_tokens: Number of tokens in the prompt.
            completion_tokens: Number of tokens in the completion.

        Returns:
            Estimated cost in USD.
        """
        # TODO: Apply current gpt-4o-mini pricing per 1K tokens for input and output.
        raise NotImplementedError

    def collect_results(
        self,
        examples: List[Dict],
        base_preds: List[List[Dict]],
        finetuned_preds: List[List[Dict]],
        gpt_preds: List[List[Dict]],
        base_latencies: List[float],
        finetuned_latencies: List[float],
        gpt_latencies: List[float],
        gpt_cost: float,
    ) -> pd.DataFrame:
        """
        Assemble a combined results DataFrame with metrics for all three systems.

        Args:
            examples: Test set examples (ground truth).
            base_preds: Base GLiNER predictions per example.
            finetuned_preds: Fine-tuned GLiNER predictions per example.
            gpt_preds: GPT predictions per example.
            base_latencies: Per-example latency in ms for base GLiNER.
            finetuned_latencies: Per-example latency in ms for fine-tuned GLiNER.
            gpt_latencies: Per-example latency in ms for GPT.
            gpt_cost: Total GPT inference cost in USD.

        Returns:
            DataFrame with columns: system, f1, precision, recall,
            p50_latency_ms, p99_latency_ms, total_cost_usd.
        """
        # TODO: Compute F1/precision/recall for each system using src.evaluation.metrics.
        # TODO: Compute latency percentiles via metrics.latency_percentiles.
        # TODO: Build and return a three-row DataFrame (one row per system).
        raise NotImplementedError

    def run(self, test_file: str = "data/splits/test.jsonl") -> pd.DataFrame:
        """
        End-to-end benchmark: load models, run inference, return results DataFrame.

        Args:
            test_file: Path to the JSONL test split.

        Returns:
            Results DataFrame from collect_results.
        """
        # TODO: Load test examples from test_file.
        # TODO: Call load_models.
        # TODO: Run all three inference methods.
        # TODO: Call collect_results and return the DataFrame.
        raise NotImplementedError
