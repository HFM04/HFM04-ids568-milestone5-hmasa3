"""Unit tests for the in-process cache."""
import asyncio
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.caching import InProcessCache, make_cache_key, should_cache


@pytest.fixture
def cache():
    return InProcessCache(max_entries=5, ttl_seconds=60)


@pytest.mark.asyncio
async def test_set_and_get(cache):
    await cache.set("key1", "value1")
    result = await cache.get("key1")
    assert result == "value1"


@pytest.mark.asyncio
async def test_miss_returns_none(cache):
    result = await cache.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_ttl_expiry():
    short_cache = InProcessCache(max_entries=10, ttl_seconds=0)
    await short_cache.set("k", "v")
    await asyncio.sleep(0.01)
    result = await short_cache.get("k")
    assert result is None


@pytest.mark.asyncio
async def test_lru_eviction(cache):
    for i in range(6):
        await cache.set(f"key{i}", f"val{i}")
    # First key should have been evicted (LRU)
    result = await cache.get("key0")
    assert result is None


@pytest.mark.asyncio
async def test_stats(cache):
    await cache.set("k", "v")
    await cache.get("k")      # hit
    await cache.get("miss")   # miss
    stats = await cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


@pytest.mark.asyncio
async def test_clear(cache):
    await cache.set("a", "1")
    await cache.set("b", "2")
    count = await cache.clear()
    assert count == 2
    assert await cache.get("a") is None


def test_cache_key_is_deterministic():
    k1 = make_cache_key("hello", "gpt2", 0.0, 50)
    k2 = make_cache_key("hello", "gpt2", 0.0, 50)
    assert k1 == k2


def test_cache_key_no_plaintext():
    key = make_cache_key("my secret prompt", "gpt2", 0.0, 50)
    assert "my secret prompt" not in key
    assert key.startswith("llm:")


def test_cache_key_differs_by_prompt():
    k1 = make_cache_key("prompt A", "gpt2", 0.0, 50)
    k2 = make_cache_key("prompt B", "gpt2", 0.0, 50)
    assert k1 != k2


def test_should_cache_only_greedy():
    assert should_cache(0.0) is True
    assert should_cache(0.5) is False
    assert should_cache(1.0) is False
