import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gliner import GLiNER
from openai import OpenAI

logger = logging.getLogger(__name__)


class NERBenchmark:
    """Runs inference with base GLiNER, fine-tuned GLiNER, and GPT-4o-mini on the test set."""

    # Fixed — ask for text and label only, compute offsets ourselves
    GPT_ANNOTATION_PROMPT = """Extract named entities from the following sports text.
Return ONLY valid JSON, no explanation, no markdown:
{{"entities": [{{"text": "<exact surface form>", "label": "<ENTITY_TYPE>"}}]}}

Entity types: PLAYER, TEAM, POSITION, STAT, INJURY, TRADE_ASSET, GAME_EVENT, VENUE, COACH, AWARD

Rules:
- Use exact surface form as it appears in the text
- STAT must include number AND metric (e.g. "32 points" not "32")
- Return empty list if no entities found

Text: {text}"""

    def __init__(
        self,
        base_model_name: str = "urchade/gliner_medium-v2.1",
        finetuned_model_path: str = "checkpoints/gliner-sports/best",
        gpt_model: str = "gpt-4o-mini",
        entity_types: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> None:
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

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load all three inference systems into memory."""
        logger.info("Loading base GLiNER...")
        self.base_gliner = GLiNER.from_pretrained(self.base_model_name)

        logger.info(f"Loading fine-tuned GLiNER from {self.finetuned_model_path}...")
        self.finetuned_gliner = GLiNER.from_pretrained(
            self.finetuned_model_path,
            load_tokenizer=True
        )

        logger.info("Initializing OpenAI client...")
        self.openai_client = OpenAI()

        logger.info("All models loaded.")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_test_examples(self, test_file: str) -> List[Dict]:
        """Load test examples from JSONL — our annotated format with text + entities."""
        examples = []
        with open(test_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        logger.info(f"Loaded {len(examples)} test examples from {test_file}")
        return examples

    # ------------------------------------------------------------------
    # Offset alignment (same as annotator — text only from GPT)
    # ------------------------------------------------------------------

    def find_entity_offsets(
        self, text: str, entity_text: str
    ) -> Optional[Tuple[int, int]]:
        """Find character offsets of entity_text in text."""
        import re
        normalized = entity_text.strip().rstrip(".,;:'\"")

        idx = text.find(normalized)
        if idx != -1:
            return idx, idx + len(normalized)

        idx = text.lower().find(normalized.lower())
        if idx != -1:
            return idx, idx + len(normalized)

        stripped = re.sub(r"'s?$", "", normalized).strip()
        if stripped != normalized:
            idx = text.lower().find(stripped.lower())
            if idx != -1:
                return idx, idx + len(stripped)

        return None

    # ------------------------------------------------------------------
    # GLiNER inference
    # ------------------------------------------------------------------

    def run_base_gliner(
        self, examples: List[Dict]
    ) -> Tuple[List[List[Dict]], List[float]]:
        """Run base GLiNER over test set, return predictions and latencies."""
        predictions = []
        latencies = []

        for example in examples:
            text = example.get("text", "")
            start = time.perf_counter()
            entities = self.base_gliner.predict_entities(
                text, self.entity_types, threshold=self.threshold
            )
            latency_ms = (time.perf_counter() - start) * 1000

            predictions.append(entities)
            latencies.append(latency_ms)

        logger.info(f"Base GLiNER: {len(predictions)} predictions, "
                   f"p50={np.percentile(latencies, 50):.1f}ms")
        return predictions, latencies

    def run_finetuned_gliner(
        self, examples: List[Dict]
    ) -> Tuple[List[List[Dict]], List[float]]:
        """Run fine-tuned GLiNER over test set."""
        predictions = []
        latencies = []

        for example in examples:
            text = example.get("text", "")
            start = time.perf_counter()
            entities = self.finetuned_gliner.predict_entities(
                text, self.entity_types, threshold=self.threshold
            )
            latency_ms = (time.perf_counter() - start) * 1000

            predictions.append(entities)
            latencies.append(latency_ms)

        logger.info(f"Fine-tuned GLiNER: {len(predictions)} predictions, "
                   f"p50={np.percentile(latencies, 50):.1f}ms")
        return predictions, latencies

    # ------------------------------------------------------------------
    # GPT inference
    # ------------------------------------------------------------------

    def run_gpt(
        self, examples: List[Dict]
    ) -> Tuple[List[List[Dict]], List[float], float]:
        """Run GPT-4o-mini over test set."""
        predictions = []
        latencies = []
        total_cost = 0.0

        for i, example in enumerate(examples):
            text = example.get("text", "")
            prompt = self.GPT_ANNOTATION_PROMPT.format(text=text)

            start = time.perf_counter()
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.gpt_model,
                    messages=[
                        {"role": "system", "content": "You are a precise NER annotator. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )
                latency_ms = (time.perf_counter() - start) * 1000

                content = response.choices[0].message.content
                parsed = json.loads(content)
                raw_entities = parsed.get("entities", [])

                # Compute offsets ourselves — never trust GPT for math
                aligned = []
                for ent in raw_entities:
                    entity_text = ent.get("text", "").strip()
                    label = ent.get("label", "").strip().upper()
                    if not entity_text or not label:
                        continue
                    offsets = self.find_entity_offsets(text, entity_text)
                    if offsets:
                        aligned.append({
                            "text": entity_text,
                            "label": label,
                            "start": offsets[0],
                            "end": offsets[1],
                        })

                predictions.append(aligned)
                latencies.append(latency_ms)

                # Track cost
                usage = response.usage
                total_cost += self.estimate_gpt_cost(
                    usage.prompt_tokens,
                    usage.completion_tokens
                )

            except Exception as e:
                logger.warning(f"GPT failed on example {i}: {e}")
                predictions.append([])
                latencies.append(0.0)

            # Rate limit buffer
            time.sleep(0.1)

        logger.info(f"GPT: {len(predictions)} predictions, "
                   f"p50={np.percentile(latencies, 50):.1f}ms, "
                   f"cost=${total_cost:.4f}")
        return predictions, latencies, total_cost

    def estimate_gpt_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """
        Estimate USD cost for GPT-4o-mini.
        Pricing: $0.150/1M input tokens, $0.600/1M output tokens
        """
        input_cost = (prompt_tokens / 1_000_000) * 0.150
        output_cost = (completion_tokens / 1_000_000) * 0.600
        return input_cost + output_cost

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        examples: List[Dict],
        predictions: List[List[Dict]],
    ) -> Dict:
        """
        Compute precision, recall, F1 against ground truth.
        Uses exact span match — both label and text must match.

        Args:
            examples: Ground truth examples with 'text' and 'entities'.
            predictions: Predicted entity lists per example.

        Returns:
            Dict with precision, recall, f1.
        """
        tp = fp = fn = 0

        for example, preds in zip(examples, predictions):
            text = example.get("text", "")
            gt_entities = example.get("entities", [])

            # Ground truth set — (text_span, label)
            gt_set = set()
            for ent in gt_entities:
                span_text = text[ent["start"]:ent["end"]].lower().strip()
                gt_set.add((span_text, ent["label"]))

            # Prediction set
            pred_set = set()
            for ent in preds:
                if "start" in ent and "end" in ent:
                    span_text = text[ent["start"]:ent["end"]].lower().strip()
                else:
                    span_text = ent.get("text", "").lower().strip()
                pred_set.add((span_text, ent["label"]))

            tp += len(gt_set & pred_set)
            fp += len(pred_set - gt_set)
            fn += len(gt_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # ------------------------------------------------------------------
    # Results assembly
    # ------------------------------------------------------------------

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
        """Assemble benchmark results into a DataFrame."""

        rows = []
        total_docs = len(examples)

        systems = [
            ("gliner_base", base_preds, base_latencies, 0.0),
            ("gliner_finetuned", finetuned_preds, finetuned_latencies, 0.0),
            ("gpt4o_mini", gpt_preds, gpt_latencies, gpt_cost),
        ]

        for name, preds, latencies, cost in systems:
            metrics = self.compute_metrics(examples, preds)
            lat = np.array(latencies)

            # Cost per 1K documents — normalize for fair comparison
            cost_per_1k = (cost / total_docs * 1000) if total_docs > 0 else 0.0

            rows.append({
                "system": name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "p50_latency_ms": round(float(np.percentile(lat, 50)), 1),
                "p99_latency_ms": round(float(np.percentile(lat, 99)), 1),
                "total_cost_usd": round(cost, 4),
                "cost_per_1k_docs": round(cost_per_1k, 4),
            })

        df = pd.DataFrame(rows)
        return df

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self, test_file: str = "data/splits/test.jsonl") -> pd.DataFrame:
        """End-to-end benchmark."""
        examples = self.load_test_examples(test_file)
        self.load_models()

        logger.info("Running base GLiNER...")
        base_preds, base_lat = self.run_base_gliner(examples)

        logger.info("Running fine-tuned GLiNER...")
        ft_preds, ft_lat = self.run_finetuned_gliner(examples)

        logger.info("Running GPT-4o-mini...")
        gpt_preds, gpt_lat, gpt_cost = self.run_gpt(examples)

        df = self.collect_results(
            examples,
            base_preds, ft_preds, gpt_preds,
            base_lat, ft_lat, gpt_lat,
            gpt_cost,
        )

        logger.info("\n" + df.to_string(index=False))
        return df