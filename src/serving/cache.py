import hashlib
from collections import OrderedDict
from typing import Any, Optional


class InMemoryCache:
    """
    LRU cache for NER extraction results.
    Key: SHA-256 of canonical request string.
    Eviction: Least recently used when max_size exceeded.
    """

    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _make_key(self, raw_key: str) -> str:
        """SHA-256 hash of raw key string."""
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def get(self, raw_key: str) -> Optional[Any]:
        """Retrieve cached value. Returns None on miss."""
        key = self._make_key(raw_key)
        if key not in self._cache:
            self._misses += 1
            return None
        # Move to end — most recently used
        self._cache.move_to_end(key)
        self._hits += 1
        return self._cache[key]

    def set(self, raw_key: str, value: Any) -> None:
        """Store value. Evicts LRU entry if at capacity."""
        key = self._make_key(raw_key)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)  # evict LRU

    def invalidate(self, raw_key: str) -> None:
        """Remove a specific key from cache."""
        key = self._make_key(raw_key)
        self._cache.pop(key, None)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a float in [0, 1]."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)