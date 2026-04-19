import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

VALID_LABELS = {
    "PLAYER", "TEAM", "POSITION", "STAT", "INJURY",
    "TRADE_ASSET", "GAME_EVENT", "VENUE", "COACH", "AWARD"
}


class AnnotationValidator:
    """Validates, filters, and splits annotated GLiNER training data."""

    def __init__(
        self,
        processed_dir: str = "data/processed",
        splits_dir: str = "data/splits",
        raw_dir: str = "data/raw",
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        seed: int = 42,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.splits_dir = Path(splits_dir)
        self.raw_dir = Path(raw_dir)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_annotations(self, filename: Optional[str] = None) -> List[Dict]:
        """
        Load all processed annotations.

        Args:
            filename: Specific JSONL file; if None loads all *.jsonl in processed_dir.

        Returns:
            Flat list of GLiNER-format annotation dicts.
        """
        if filename:
            paths = [self.processed_dir / filename]
        else:
            paths = list(self.processed_dir.glob("*.jsonl"))

        examples = []
        for path in paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        examples.append(json.loads(line))
            logger.info(f"Loaded {len(examples)} examples from {path}")

        logger.info(f"Total loaded: {len(examples)} examples")
        return examples

    def load_weak_labels(self) -> Dict[str, Dict]:
        """
        Load ESPN weak labels from raw JSONL files.
        Returns a dict mapping article body snippet → weak_labels dict.
        We use first 100 chars as a loose key since sentences don't have article IDs.

        Returns:
            Dict mapping body_prefix → weak_labels.
        """
        weak_map = {}
        for path in self.raw_dir.glob("*.jsonl"):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    article = json.loads(line)
                    body = article.get("body", "")
                    weak_labels = article.get("weak_labels", {})
                    if body and weak_labels:
                        # Key by first 80 chars — enough to match sentences back to articles
                        weak_map[body[:80]] = weak_labels
        return weak_map

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------

    def flag_low_quality(
        self,
        annotations: List[Dict],
        min_entities: int = 1,
        max_entities: int = 50,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Separate valid from flagged annotations.

        Flags:
        - Zero entities
        - Entity count outside [min_entities, max_entities]
        - Invalid label not in VALID_LABELS
        - Entity offsets outside text bounds
        - Entity text doesn't match text[start:end]

        Args:
            annotations: Full list of GLiNER-format examples.
            min_entities: Minimum entity count threshold.
            max_entities: Maximum entity count threshold.

        Returns:
            (valid, flagged) tuple of lists.
        """
        valid = []
        flagged = []

        for example in annotations:
            text = example.get("text", "")
            entities = example.get("entities", [])
            reasons = []

            # Check entity count bounds
            if len(entities) < min_entities:
                reasons.append(f"too_few_entities ({len(entities)})")

            if len(entities) > max_entities:
                reasons.append(f"too_many_entities ({len(entities)})")

            for ent in entities:
                start = ent.get("start", -1)
                end = ent.get("end", -1)
                label = ent.get("label", "")

                # Bounds check
                if start < 0 or end > len(text) or start >= end:
                    reasons.append(f"invalid_offsets ({start},{end})")
                    break

                # Label validity
                if label not in VALID_LABELS:
                    reasons.append(f"invalid_label ({label})")
                    break

                # Alignment check — does text[start:end] make sense?
                extracted = text[start:end]
                if len(extracted.strip()) == 0:
                    reasons.append("empty_entity_text")
                    break

            if reasons:
                example["_flag_reasons"] = reasons
                flagged.append(example)
            else:
                valid.append(example)

        logger.info(
            f"Quality filter: {len(valid)} valid, {len(flagged)} flagged "
            f"({len(flagged)/len(annotations)*100:.1f}% flagged)"
        )
        return valid, flagged

    def check_weak_label_coverage(
        self,
        annotations: List[Dict],
        raw_articles: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Validate annotations against ESPN weak labels.
        For each article, check if known teams/players appear in annotations.

        Args:
            annotations: List of GLiNER-format examples.
            raw_articles: Raw article dicts with weak_labels. Loads from disk if None.

        Returns:
            Coverage report dict.
        """
        if raw_articles is None:
            raw_articles = []
            for path in self.raw_dir.glob("*.jsonl"):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            raw_articles.append(json.loads(line))

        # Build set of all annotated entity texts
        annotated_entities = set()
        for example in annotations:
            for ent in example.get("entities", []):
                annotated_entities.add(ent.get("label", ""))

        # Check coverage per article
        total_articles = 0
        covered_articles = 0
        missed_entities = []

        for article in raw_articles:
            weak = article.get("weak_labels", {})
            known_teams = weak.get("teams", [])
            known_players = weak.get("players", [])
            all_known = known_teams + known_players

            if not all_known:
                continue

            total_articles += 1
            body = article.get("body", "")

            # Check if any annotation from this article captured a known entity
            article_annotations = [
                ex for ex in annotations
                if ex.get("text", "") in body or body[:50] in ex.get("text", "")
            ]

            found_any = False
            for known in all_known:
                for ex in article_annotations:
                    for ent in ex.get("entities", []):
                        if known.lower() in ex["text"][ent["start"]:ent["end"]].lower():
                            found_any = True
                            break

            if found_any:
                covered_articles += 1
            else:
                missed_entities.extend(all_known)

        coverage_rate = covered_articles / total_articles if total_articles > 0 else 0

        report = {
            "total_articles_with_weak_labels": total_articles,
            "covered_articles": covered_articles,
            "coverage_rate": round(coverage_rate, 3),
            "sample_missed_entities": list(set(missed_entities))[:10],
        }

        logger.info(f"Weak label coverage: {coverage_rate:.1%} ({covered_articles}/{total_articles})")
        return report

    def compute_inter_annotator_agreement(
        self,
        annotations_a: List[Dict],
        annotations_b: List[Dict],
    ) -> float:
        """
        Compute Cohen's Kappa between two annotators on the same texts.
        Converts span annotations to token-level label sequences for comparison.

        Args:
            annotations_a: First annotator's examples.
            annotations_b: Second annotator's examples.

        Returns:
            Cohen's Kappa coefficient.
        """
        # Align by text
        text_to_b = {ex["text"]: ex for ex in annotations_b}

        labels_a = []
        labels_b = []

        for ex_a in annotations_a:
            text = ex_a["text"]
            if text not in text_to_b:
                continue

            ex_b = text_to_b[text]
            tokens = text.split()

            # Build token-level label arrays for each annotator
            seq_a = self._spans_to_token_labels(tokens, ex_a.get("entities", []), text)
            seq_b = self._spans_to_token_labels(tokens, ex_b.get("entities", []), text)

            labels_a.extend(seq_a)
            labels_b.extend(seq_b)

        if not labels_a:
            logger.warning("No overlapping texts found for Kappa computation")
            return 0.0

        kappa = cohen_kappa_score(labels_a, labels_b)
        logger.info(f"Cohen's Kappa: {kappa:.3f}")
        return kappa

    def _spans_to_token_labels(
        self,
        tokens: List[str],
        entities: List[Dict],
        text: str,
    ) -> List[str]:
        """
        Convert character-level spans to token-level BIO label sequence.

        Args:
            tokens: Whitespace-split tokens.
            entities: List of entity dicts with start/end/label.
            text: Original text string.

        Returns:
            List of label strings, one per token (O for non-entity).
        """
        labels = ["O"] * len(tokens)

        # Map character offset to token index
        char_to_token = {}
        char_pos = 0
        for i, token in enumerate(tokens):
            for _ in token:
                char_to_token[char_pos] = i
                char_pos += 1
            char_pos += 1  # space

        for ent in entities:
            start = ent.get("start", -1)
            end = ent.get("end", -1)
            label = ent.get("label", "O")

            if start in char_to_token:
                token_idx = char_to_token[start]
                labels[token_idx] = label

        return labels

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self, annotations: List[Dict]) -> Dict:
        """
        Compute summary statistics for a set of annotations.

        Args:
            annotations: List of GLiNER-format examples.

        Returns:
            Summary dict with counts, entity stats, and label distribution.
        """
        total_examples = len(annotations)
        entity_counts = [len(ex.get("entities", [])) for ex in annotations]
        total_entities = sum(entity_counts)

        label_counts: Counter = Counter()
        for ex in annotations:
            for ent in ex.get("entities", []):
                label_counts[ent.get("label", "UNKNOWN")] += 1

        report = {
            "total_examples": total_examples,
            "total_entities": total_entities,
            "avg_entities_per_example": round(np.mean(entity_counts), 2),
            "std_entities_per_example": round(np.std(entity_counts), 2),
            "per_label_counts": dict(label_counts.most_common()),
        }

        logger.info(f"Report: {total_examples} examples, {total_entities} entities")
        logger.info(f"Label distribution: {dict(label_counts.most_common())}")
        return report

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def create_splits(
        self, annotations: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Shuffle and split annotations into train/val/test.

        Args:
            annotations: Full validated annotation list.

        Returns:
            (train, val, test) tuple.
        """
        random.seed(self.seed)
        shuffled = annotations.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]

        logger.info(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def save_splits(
        self,
        train: List[Dict],
        val: List[Dict],
        test: List[Dict],
    ) -> None:
        """
        Write splits to JSONL files.

        Args:
            train: Training examples.
            val: Validation examples.
            test: Test examples.
        """
        for name, split in [("train", train), ("val", val), ("test", test)]:
            path = self.splits_dir / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for example in split:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(split)} examples to {path}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """
        Full validation and splitting pipeline.

        Returns:
            Validation report dict.
        """
        # Load all annotations
        annotations = self.load_annotations()
        logger.info(f"Loaded {len(annotations)} total examples")

        # Quality filtering
        valid, flagged = self.flag_low_quality(annotations)
        logger.info(f"After filtering: {len(valid)} valid examples")

        # Weak label coverage check
        coverage = self.check_weak_label_coverage(valid)

        # Generate report on valid data
        report = self.generate_report(valid)
        report["weak_label_coverage"] = coverage
        report["flagged_count"] = len(flagged)

        # Create and save splits
        train, val, test = self.create_splits(valid)
        self.save_splits(train, val, test)

        return report