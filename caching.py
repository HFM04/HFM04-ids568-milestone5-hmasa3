"""
In-process LRU cache with TTL expiration for LLM inference responses.

Privacy design:
  - Cache keys are SHA-256 hashes of (prompt + model parameters).
  - No plaintext prompts, user IDs, or PII are ever stored.
  - TTL ensures stale responses expire automatically.
  - Max-entry limit with LRU eviction prevents unbounded memory growth.
"""

import hashlib
import json
import time
import asyncio
from collections import OrderedDict
from typing import Optional
from src.config import settings


class InProcessCache:
    """
    Thread-safe, async-compatible LRU cache with per-entry TTL.

    Key design:
      - Uses an OrderedDict for O(1) LRU tracking.
      - asyncio.Lock protects all mutations (get, set, evict).
      - Entries store (value, expiry_timestamp) tuples.
    """

    def __init__(self, max_entries: int, ttl_seconds: int):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._lock = asyncio.Lock()

        # Metrics counters
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    # ── Public API ──────────────────────────────────────────────────────────

    async def get(self, key: str) -> Optional[str]:
        """Return cached value or None if missing/expired."""
        async with self._lock:
            if key not in self._store:
                self._misses += 1
                return None

            value, expiry = self._store[key]

            # Check TTL expiry
            if time.monotonic() > expiry:
                del self._store[key]
                self._expirations += 1
                self._misses += 1
                return None

            # Move to end = most recently used
            self._store.move_to_end(key)
            self._hits += 1
            return value

    async def set(self, key: str, value: str) -> None:
        """Store a value with TTL. Evicts LRU entry if at capacity."""
        async with self._lock:
            expiry = time.monotonic() + self.ttl_seconds

            if key in self._store:
                # Update existing entry
                self._store[key] = (value, expiry)
                self._store.move_to_end(key)
            else:
                # Evict LRU entry if at capacity
                if len(self._store) >= self.max_entries:
                    self._store.popitem(last=False)  # Remove oldest
                    self._evictions += 1
                self._store[key] = (value, expiry)

    async def invalidate(self, key: str) -> bool:
        """Manually remove an entry. Returns True if it existed."""
        async with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def clear(self) -> int:
        """Remove all entries. Returns count of cleared entries."""
        async with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    async def purge_expired(self) -> int:
        """Scan and remove all expired entries. Returns count removed."""
        async with self._lock:
            now = time.monotonic()
            expired_keys = [
                k for k, (_, expiry) in self._store.items()
                if now > expiry
            ]
            for k in expired_keys:
                del self._store[k]
                self._expirations += 1
            return len(expired_keys)

    # ── Metrics ─────────────────────────────────────────────────────────────

    async def stats(self) -> dict:
        """Return cache statistics snapshot."""
        async with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._store),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "evictions": self._evictions,
                "expirations": self._expirations,
                "total_requests": total,
            }

    def reset_stats(self) -> None:
        """Reset hit/miss counters (useful between benchmark runs)."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0


# ── Cache Key Construction ──────────────────────────────────────────────────

def make_cache_key(prompt: str, model: str, temperature: float, max_new_tokens: int) -> str:
    """
    Build a privacy-preserving cache key.

    Rules:
      - Only deterministic requests (temperature == 0.0) should be cached.
      - The key is a SHA-256 hash — no plaintext content is stored as the key.
      - User identifiers are NEVER included.

    Args:
        prompt: The user prompt text.
        model: Model name string.
        temperature: Sampling temperature.
        max_new_tokens: Generation length cap.

    Returns:
        Hex string cache key prefixed with "llm:".
    """
    key_data = {
        "prompt": prompt.strip(),          # Normalise whitespace only
        "model": model,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
    }
    content = json.dumps(key_data, sort_keys=True)
    digest = hashlib.sha256(content.encode()).hexdigest()
    return f"llm:{digest}"


def should_cache(temperature: float) -> bool:
    """
    Only cache deterministic (greedy) generations.
    Non-zero temperature produces different outputs each call — not cacheable.
    """
    return temperature == 0.0


# ── Singleton Cache Instance ─────────────────────────────────────────────────
cache = InProcessCache(
    max_entries=settings.cache_max_entries,
    ttl_seconds=settings.cache_ttl_seconds,
)
