"""Embedding cache with LRU eviction"""
from collections import OrderedDict
from typing import Optional, Tuple
from src.utils import get_logger

logger = get_logger()


class EmbeddingCache:
    """LRU cache for embeddings (50% speed boost potential)"""

    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, list] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, text: str) -> Optional[list]:
        """Get embedding from cache (O(1) lookup)"""
        key = self._hash(text)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)  # Mark as recently used
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, text: str, embedding: list) -> None:
        """Store embedding with LRU eviction"""
        key = self._hash(text)
        self.cache[key] = embedding
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            old_key, _ = self.cache.popitem(last=False)  # Remove oldest

    def clear(self) -> None:
        """Clear all cached embeddings"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Cache efficiency metric (0-1)"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

    @property
    def size(self) -> int:
        """Current cache size"""
        return len(self.cache)

    @staticmethod
    def _hash(text: str) -> str:
        """Hash text for cache key"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
