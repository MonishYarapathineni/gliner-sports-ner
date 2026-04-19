import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
import openai

logger = logging.getLogger(__name__)


class GPTAnnotator:
    """Annotates raw sports text using GPT-4o-mini to produce GLiNER training data."""

    # Removed start/end from schema — we compute offsets ourselves
    ANNOTATION_PROMPT_TEMPLATE = """You are an expert sports NER annotator. Given the following sports text, identify and label all named entities.

Entity types:
- PLAYER: Individual athlete names (e.g. "Donovan Mitchell", "Patrick Mahomes")
- TEAM: Sports team names (e.g. "Cleveland Cavaliers", "Kansas City Chiefs")
- POSITION: Player positions (e.g. "point guard", "quarterback", "striker")
- STAT: Numerical statistics (e.g. "32 points", "4 touchdowns", ".342 batting average")
- INJURY: Injury descriptions (e.g. "torn ACL", "hamstring strain", "concussion protocol")
- TRADE_ASSET: Draft picks or trade pieces (e.g. "first-round pick", "salary cap space")
- GAME_EVENT: Notable in-game events (e.g. "buzzer beater", "walk-off homer", "red card")
- VENUE: Stadium or arena names (e.g. "Madison Square Garden", "Lambeau Field")
- COACH: Coach or staff names (e.g. "Erik Spoelstra", "Nick Saban")
- AWARD: Awards or accolades (e.g. "MVP", "Cy Young", "Golden Boot")

Rules:
- Only annotate entities you are certain about
- Use the exact surface form as it appears in the text
- Do not annotate pronouns or generic references
- STAT must include the number AND the metric (e.g. "32 points" not just "32")

Return ONLY valid JSON, no explanation, no markdown:
{{"entities": [{{"text": "<exact surface form>", "label": "<ENTITY_TYPE>"}}]}}

Text:
{text}"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        max_retries: int = 3,
        requests_per_minute: int = 500,
    ) -> None:
        self.model = model
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.min_interval = 60.0 / requests_per_minute
        self.client = OpenAI()

    # ------------------------------------------------------------------
    # Text preprocessing
    # ------------------------------------------------------------------

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split article body into sentences suitable for GLiNER training.
        Targets 1-2 sentences per chunk, max ~300 chars.

        Args:
            text: Full article body text.

        Returns:
            List of sentence strings.
        """
        import re
        # Split on sentence boundaries
        raw = re.split(r'(?<=[.!?])\s+', text.strip())

        sentences = []
        buffer = ""

        for sentence in raw:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence keeps us under 300 chars, buffer it
            if len(buffer) + len(sentence) < 300:
                buffer = (buffer + " " + sentence).strip()
            else:
                if buffer:
                    sentences.append(buffer)
                buffer = sentence

        if buffer:
            sentences.append(buffer)

        # Filter out very short chunks — not useful for NER
        return [s for s in sentences if len(s) > 30]

    # ------------------------------------------------------------------
    # Offset computation — we do the math, not GPT
    # ------------------------------------------------------------------

    def find_entity_offsets(
        self, text: str, entity_text: str
    ) -> Optional[tuple]:
        """
        Find the character-level start and end offsets of an entity in text.
        Uses case-insensitive search with punctuation normalization.

        Args:
            text: The sentence text.
            entity_text: The entity surface form returned by GPT.

        Returns:
            (start, end) char offsets inclusive, or None if not found.
        """
        import re

        # Normalize both for matching — strip trailing punctuation
        normalized_text = text
        normalized_entity = entity_text.strip().rstrip(".,;:'\"")

        # Try exact match first
        idx = normalized_text.find(normalized_entity)
        if idx != -1:
            return idx, idx + len(normalized_entity)

        # Try case-insensitive match
        lower_text = normalized_text.lower()
        lower_entity = normalized_entity.lower()
        idx = lower_text.find(lower_entity)
        if idx != -1:
            return idx, idx + len(normalized_entity)

        # Try stripping possessives from entity ("Cavaliers'" -> "Cavaliers")
        stripped = re.sub(r"'s?$", "", normalized_entity).strip()
        if stripped != normalized_entity:
            idx = normalized_text.lower().find(stripped.lower())
            if idx != -1:
                return idx, idx + len(stripped)

        logger.debug(f"Could not align entity '{entity_text}' in text")
        return None

    # ------------------------------------------------------------------
    # Core annotation methods
    # ------------------------------------------------------------------

    def load_raw_articles(self, filename: str) -> List[Dict]:
        """
        Load articles from a JSONL file in the raw data directory.

        Args:
            filename: JSONL filename (e.g. 'nba_raw.jsonl').

        Returns:
            List of article dicts.
        """
        path = self.raw_dir / filename
        articles = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    articles.append(json.loads(line))
        logger.info(f"Loaded {len(articles)} articles from {path}")
        return articles

    def build_prompt(self, text: str) -> str:
        """
        Format the annotation prompt with the provided text.

        Args:
            text: Sentence or chunk of article text.

        Returns:
            Formatted prompt string.
        """
        return self.ANNOTATION_PROMPT_TEMPLATE.format(text=text)

    def call_gpt(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Call GPT-4o-mini and return parsed JSON entity list.
        Retries on rate limit errors and JSON decode errors.

        Args:
            text: Sentence text to annotate.

        Returns:
            Parsed dict with 'entities' key, or None if all retries fail.
        """
        prompt = self.build_prompt(text)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise NER annotator. Return only valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.0,  # deterministic — critical for annotation consistency
                    max_tokens=1000,
                    response_format={"type": "json_object"},  # force JSON output
                )

                content = response.choices[0].message.content
                parsed = json.loads(content)

                # Validate schema
                if "entities" not in parsed:
                    logger.warning("GPT response missing 'entities' key")
                    return None

                time.sleep(self.min_interval)
                return parsed

            except openai.RateLimitError:
                wait = (2 ** attempt) * 5
                logger.warning(f"Rate limited. Waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode failed (attempt {attempt+1}): {e}")
                time.sleep(1)

            except openai.OpenAIError as e:
                logger.error(f"OpenAI error: {e}")
                return None

        logger.error(f"All retries exhausted for text: {text[:50]}...")
        return None

    def convert_to_gliner_format(
        self, text: str, gpt_response: Dict[str, Any]
    ) -> Dict:
        """
        Convert GPT annotation response into GLiNER training format.
        Computes character offsets ourselves — never trusts GPT for math.

        GLiNER format:
        {
            "text": "sentence text",
            "entities": [{"start": 0, "end": 16, "label": "PLAYER"}]
        }

        Args:
            text: The sentence that was annotated.
            gpt_response: Parsed JSON from call_gpt.

        Returns:
            GLiNER-format training example dict.
        """
        raw_entities = gpt_response.get("entities", [])
        aligned_entities = []

        seen_spans = set()  # prevent duplicate spans

        for ent in raw_entities:
            entity_text = ent.get("text", "").strip()
            label = ent.get("label", "").strip().upper()

            if not entity_text or not label:
                continue

            offsets = self.find_entity_offsets(text, entity_text)
            if offsets is None:
                logger.debug(f"Skipping unaligned entity: '{entity_text}'")
                continue

            start, end = offsets

            # Skip duplicate spans
            span_key = (start, end, label)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)

            aligned_entities.append({
                "start": start,
                "end": end,
                "label": label,
            })

        return {
            "text": text,
            "entities": aligned_entities,
        }

    def annotate_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Annotate all articles — splits into sentences, annotates each,
        returns GLiNER training examples.

        Args:
            articles: List of raw article dicts with 'body' key.

        Returns:
            List of GLiNER training example dicts.
        """
        all_examples = []
        total_sentences = 0
        failed = 0

        for i, article in enumerate(articles):
            body = article.get("body", "")
            if not body:
                continue

            sentences = self.split_into_sentences(body)
            total_sentences += len(sentences)

            for sentence in sentences:
                gpt_response = self.call_gpt(sentence)

                if gpt_response is None:
                    failed += 1
                    continue

                example = self.convert_to_gliner_format(sentence, gpt_response)

                # Only keep examples that have at least one entity
                if example["entities"]:
                    all_examples.append(example)

            if (i + 1) % 10 == 0:
                logger.info(
                    f"Progress: {i+1}/{len(articles)} articles | "
                    f"{len(all_examples)} examples collected | "
                    f"{failed} failed"
                )

        logger.info(
            f"Annotation complete: {len(all_examples)} examples from "
            f"{total_sentences} sentences | {failed} failed calls"
        )
        return all_examples

    def save_annotated(self, examples: List[Dict], filename: str) -> Path:
        """
        Save GLiNER training examples to JSONL.

        Args:
            examples: List of GLiNER training example dicts.
            filename: Output filename.

        Returns:
            Path to written file.
        """
        out_path = self.processed_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(examples)} examples to {out_path}")
        return out_path

    def run(self, raw_filenames: Optional[List[str]] = None) -> None:
        """
        Full annotation pipeline over all raw JSONL files.

        Args:
            raw_filenames: Files to process; defaults to all *.jsonl in raw_dir.
        """
        if raw_filenames is None:
            raw_filenames = [p.name for p in self.raw_dir.glob("*.jsonl")
                           if p.name != "progress.json"]

        logger.info(f"Annotating {len(raw_filenames)} files: {raw_filenames}")

        for filename in raw_filenames:
            logger.info(f"--- Annotating {filename} ---")
            articles = self.load_raw_articles(filename)
            examples = self.annotate_articles(articles)
            out_name = filename.replace("_raw.jsonl", "_annotated.jsonl")
            self.save_annotated(examples, out_name)