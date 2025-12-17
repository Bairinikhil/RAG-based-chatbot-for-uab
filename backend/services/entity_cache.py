"""
Entity Extraction Caching Layer
Provides in-memory caching with TTL for entity extraction results
"""

import time
import hashlib
import logging
from typing import Optional, Dict, Any
from collections import OrderedDict
from threading import Lock

from models.entities import ExtractionResult

logger = logging.getLogger(__name__)


class EntityCache:
    """
    LRU cache with TTL for entity extraction results
    Thread-safe implementation
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize cache

        Args:
            max_size: Maximum number of entries (LRU eviction)
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

        logger.info(f"EntityCache initialized (max_size={max_size}, ttl={ttl_seconds}s)")

    def _generate_key(self, query: str, use_llm: bool) -> str:
        """Generate cache key from query and options"""
        # Normalize query
        normalized = query.lower().strip()
        # Include LLM flag in key
        cache_input = f"{normalized}|llm={use_llm}"
        # Generate hash
        return hashlib.md5(cache_input.encode()).hexdigest()

    def get(self, query: str, use_llm: bool = True) -> Optional[ExtractionResult]:
        """
        Get cached extraction result

        Args:
            query: User query
            use_llm: Whether LLM fallback was enabled

        Returns:
            ExtractionResult if found and not expired, None otherwise
        """
        key = self._generate_key(query, use_llm)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]
            timestamp = entry["timestamp"]
            result = entry["result"]

            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                logger.debug(f"Cache entry expired for query: {query[:50]}")
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1

            logger.debug(f"Cache hit for query: {query[:50]}")
            return result

    def set(self, query: str, result: ExtractionResult, use_llm: bool = True):
        """
        Cache extraction result

        Args:
            query: User query
            result: ExtractionResult to cache
            use_llm: Whether LLM fallback was enabled
        """
        key = self._generate_key(query, use_llm)

        with self._lock:
            # If at capacity, remove oldest (LRU)
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Cache full, evicted oldest entry (size={self.max_size})")

            # Add/update entry
            self._cache[key] = {
                "timestamp": time.time(),
                "result": result
            }

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            logger.debug(f"Cached result for query: {query[:50]} (size={len(self._cache)})")

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")

    def remove_expired(self) -> int:
        """
        Remove all expired entries

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        removed = 0

        with self._lock:
            keys_to_remove = []
            for key, entry in self._cache.items():
                if current_time - entry["timestamp"] > self.ttl_seconds:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._cache[key]
                removed += 1

        if removed > 0:
            logger.info(f"Removed {removed} expired cache entries")

        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total_requests,
                "hit_rate_percent": round(hit_rate, 2),
                "ttl_seconds": self.ttl_seconds
            }

    def reset_stats(self):
        """Reset hit/miss counters"""
        with self._lock:
            self._hits = 0
            self._misses = 0
            logger.info("Cache statistics reset")


# Global cache instance
_entity_cache_instance: Optional[EntityCache] = None


def get_entity_cache() -> Optional[EntityCache]:
    """Get global entity cache instance"""
    return _entity_cache_instance


def set_entity_cache(cache: EntityCache):
    """Set global entity cache instance"""
    global _entity_cache_instance
    _entity_cache_instance = cache
    logger.info("Global entity cache instance set")


def init_entity_cache(max_size: int = 1000, ttl_seconds: int = 3600) -> EntityCache:
    """
    Initialize and set global entity cache

    Args:
        max_size: Maximum cache size
        ttl_seconds: Time-to-live for entries

    Returns:
        EntityCache instance
    """
    cache = EntityCache(max_size=max_size, ttl_seconds=ttl_seconds)
    set_entity_cache(cache)
    return cache
