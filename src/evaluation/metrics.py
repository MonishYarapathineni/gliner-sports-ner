from typing import Dict, List, Tuple

import numpy as np


def exact_match_scores(
    references: List[List[Dict]], predictions: List[List[Dict]]
) -> Tuple[float, float, float]:
    """
    Compute exact-match precision, recall, and F1 over a full dataset.

    A predicted span is a true positive only if text, start, end, and label
    all exactly match a reference span.

    Args:
        references: Ground-truth entity lists per example.
                    Each entity dict must have keys: start, end, label.
        predictions: Predicted entity lists per example, same structure.

    Returns:
        Tuple of (precision, recall, f1) as floats in [0, 1].
    """
    # TODO: Iterate parallel (refs, preds) pairs.
    # TODO: Convert each entity to a frozenset key (start, end, label).
    # TODO: Accumulate true positives, false positives, false negatives globally.
    # TODO: Compute and return precision, recall, F1.
    raise NotImplementedError


def partial_match_scores(
    references: List[List[Dict]], predictions: List[List[Dict]]
) -> Tuple[float, float, float]:
    """
    Compute partial-match (span-overlap) precision, recall, and F1.

    A prediction counts as a match if it overlaps with a reference span of the
    same label and the overlap / union ratio exceeds 0.5.

    Args:
        references: Ground-truth entity lists per example.
        predictions: Predicted entity lists per example.

    Returns:
        Tuple of (precision, recall, f1).
    """
    # TODO: For each (pred, ref) pair, compute overlap ratio.
    # TODO: Use greedy matching (sort by overlap ratio, assign 1-to-1).
    # TODO: Accumulate TP/FP/FN and return P/R/F1.
    raise NotImplementedError


def per_entity_type_scores(
    references: List[List[Dict]],
    predictions: List[List[Dict]],
    entity_types: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute exact-match F1, precision, and recall broken down by entity type.

    Args:
        references: Ground-truth entity lists per example.
        predictions: Predicted entity lists per example.
        entity_types: List of label strings to evaluate.

    Returns:
        Dict mapping each entity type to {"precision": float, "recall": float, "f1": float}.
    """
    # TODO: For each entity_type, filter references and predictions to that label.
    # TODO: Call exact_match_scores on the filtered lists.
    # TODO: Build and return the nested dict.
    raise NotImplementedError


def latency_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    """
    Compute p50, p90, and p99 latency percentiles.

    Args:
        latencies_ms: List of per-example latency measurements in milliseconds.

    Returns:
        Dict with keys "p50", "p90", "p99" mapping to float millisecond values.
    """
    # TODO: Use np.percentile to compute 50th, 90th, 99th percentiles.
    # TODO: Return {"p50": ..., "p90": ..., "p99": ...}.
    raise NotImplementedError


def cost_per_1k_documents(total_cost_usd: float, num_documents: int) -> float:
    """
    Normalize total inference cost to a per-1K-document rate.

    Args:
        total_cost_usd: Cumulative cost in USD for all documents.
        num_documents: Number of documents processed.

    Returns:
        Cost in USD per 1,000 documents.
    """
    # TODO: Return (total_cost_usd / num_documents) * 1000.
    raise NotImplementedError
