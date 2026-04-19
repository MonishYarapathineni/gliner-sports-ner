import logging
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    BOUNDARY_ERROR = "boundary_error"      # Correct label, wrong span boundaries
    LABEL_ERROR = "label_error"            # Correct span, wrong label
    MISSED_ENTITY = "missed_entity"        # Reference entity not predicted
    FALSE_POSITIVE = "false_positive"      # Predicted entity absent from references


class ErrorAnalyzer:
    """Categorizes and surfaces NER prediction errors for qualitative analysis."""

    def __init__(self, entity_types: List[str]) -> None:
        """
        Args:
            entity_types: List of valid entity type labels.
        """
        self.entity_types = entity_types

    def categorize_errors(
        self,
        references: List[List[Dict]],
        predictions: List[List[Dict]],
        texts: List[str],
    ) -> List[Dict]:
        """
        Classify every prediction error into one of the four ErrorType categories.

        Args:
            references: Ground-truth entity lists per example.
            predictions: Predicted entity lists per example.
            texts: Raw text strings corresponding to each example.

        Returns:
            List of error dicts, each with keys: text, reference_entity,
            predicted_entity, error_type (ErrorType), and entity_type.
        """
        # TODO: For each (ref_entities, pred_entities, text) triple:
        #   - Match preds to refs greedily by span overlap.
        #   - Unmatched preds → FALSE_POSITIVE.
        #   - Unmatched refs → MISSED_ENTITY.
        #   - Matched pairs with same label but different boundaries → BOUNDARY_ERROR.
        #   - Matched pairs with same boundaries but different label → LABEL_ERROR.
        # TODO: Collect and return all error dicts.
        raise NotImplementedError

    def error_distribution_by_type(self, errors: List[Dict]) -> pd.DataFrame:
        """
        Count errors broken down by entity_type and error_type.

        Args:
            errors: Error dicts from categorize_errors.

        Returns:
            DataFrame with index=entity_type, columns=ErrorType values, values=counts.
        """
        # TODO: Pivot errors into a DataFrame indexed by entity_type × error_type.
        raise NotImplementedError

    def top_failure_examples(
        self, errors: List[Dict], error_type: ErrorType, top_k: int = 10
    ) -> List[Dict]:
        """
        Return the top-k most representative failure examples for a given error type.

        Args:
            errors: Error dicts from categorize_errors.
            error_type: The ErrorType category to filter on.
            top_k: Number of examples to return.

        Returns:
            List of up to top_k error dicts for the specified error_type,
            ordered by entity_type frequency (most common first).
        """
        # TODO: Filter errors by error_type.
        # TODO: Sort by entity_type frequency (most common label first).
        # TODO: Return first top_k results.
        raise NotImplementedError

    def summarize(self, errors: List[Dict]) -> Dict:
        """
        Produce a high-level error summary dictionary.

        Args:
            errors: Error dicts from categorize_errors.

        Returns:
            Dict with total_errors, per_error_type_counts, and
            per_entity_type_error_rate.
        """
        # TODO: Count total errors and break down by ErrorType.
        # TODO: Compute per entity_type error rate (errors / total entities of that type).
        raise NotImplementedError
