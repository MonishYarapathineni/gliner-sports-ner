import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class InMemoryCache:
    """Thread-safe LRU cache for NER extraction results with hit-rate tracking."""

    def __init__(self, max_size: int = 1024) -> None:
        """
        Args:
            max_size: Maximum number of entries before LRU eviction begins.
        """
        self.max_size = max_size
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    def _make_key(self, text: str, entity_types: list, threshold: float) -> str:
        """
        Build a deterministic cache key from the extraction request parameters.

        Args:
            text: Input text string.
            entity_types: Sorted list of requested entity type labels.
            threshold: Confidence threshold value.

        Returns:
            Hex digest string usable as a cache key.
        """
        # TODO: Serialize (text, sorted(entity_types), threshold) to a canonical JSON string.
        # TODO: Return hashlib.sha256(canonical.encode()).hexdigest().
        raise NotImplementedError

    def get(self, text: str, entity_types: list, threshold: float) -> Optional[Any]:
        """
        Retrieve a cached extraction result, or return None on a cache miss.

        Moves the accessed key to the end of the OrderedDict (most recently used).

        Args:
            text: Input text string.
            entity_types: List of entity type labels from the request.
            threshold: Confidence threshold from the request.

        Returns:
            Cached result, or None if not present.
        """
        # TODO: Compute key via _make_key.
        # TODO: If present, move to end, increment _hits, return value.
        # TODO: Otherwise increment _misses and return None.
        raise NotImplementedError

    def set(self, text: str, entity_types: list, threshold: float, value: Any) -> None:
        """
        Insert a result into the cache, evicting the LRU entry if at capacity.

        Args:
            text: Input text string.
            entity_types: List of entity type labels.
            threshold: Confidence threshold.
            value: Extraction result to cache.
        """
        # TODO: Compute key via _make_key.
        # TODO: If key already exists, move to end (update).
        # TODO: If at capacity, pop the first (LRU) item.
        # TODO: Insert key → value at the end.
        raise NotImplementedError

    def invalidate(self, text: str, entity_types: list, threshold: float) -> bool:
        """
        Remove a specific entry from the cache if it exists.

        Args:
            text: Input text string.
            entity_types: List of entity type labels.
            threshold: Confidence threshold.

        Returns:
            True if the entry was found and removed, False otherwise.
        """
        # TODO: Compute key and pop from _store if present.
        raise NotImplementedError

    def clear(self) -> None:
        """Evict all entries and reset hit/miss counters."""
        # TODO: Clear _store, reset _hits and _misses to 0.
        raise NotImplementedError

    @property
    def hit_rate(self) -> float:
        """
        Fraction of lookups that resulted in a cache hit.

        Returns:
            Float in [0, 1], or 0.0 if no lookups have been made.
        """
        # TODO: Return _hits / (_hits + _misses) or 0.0.
        raise NotImplementedError

    @property
    def stats(self) -> Dict[str, Any]:
        """
        Return a snapshot of cache statistics.

        Returns:
            Dict with keys: size, max_size, hits, misses, hit_rate.
        """
        # TODO: Return dict of current stats.
        raise NotImplementedError
