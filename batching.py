"""
Dynamic request batching for LLM inference.

Strategy: Hybrid trigger — process a batch when EITHER:
  1. max_batch_size requests have accumulated, OR
  2. batch_timeout_ms milliseconds have elapsed since the first request arrived.

Concurrency safety:
  - asyncio.Lock protects the pending-request queue.
  - Each request receives an asyncio.Future; the batcher resolves it when done.
  - Background timeout processor runs as a long-lived asyncio Task.
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    """Represents one queued inference request."""
    prompt: str
    max_new_tokens: int
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass
class BatchMetrics:
    """Collected during a single batch execution."""
    batch_size: int
    queue_wait_ms: float       # Time from first enqueue to batch start
    inference_ms: float        # Time spent inside the model
    total_ms: float            # Wall time for the whole batch
    triggered_by: str          # "size" | "timeout"


class DynamicBatcher:
    """
    Collects concurrent requests and dispatches them as a single batch
    to the underlying inference function.

    Usage:
        batcher = DynamicBatcher(inference_fn)
        await batcher.start()
        result = await batcher.submit("Hello world", max_new_tokens=50)
        await batcher.stop()
    """

    def __init__(self, inference_fn, max_batch_size: int = None,
                 batch_timeout_ms: float = None):
        """
        Args:
            inference_fn: Async callable (prompts: list[str], max_new_tokens: int)
                          -> list[str].  Called with the full batch at once.
            max_batch_size: Override config default.
            batch_timeout_ms: Override config default.
        """
        self._inference_fn = inference_fn
        self.max_batch_size = max_batch_size or settings.max_batch_size
        self.batch_timeout_ms = batch_timeout_ms or settings.batch_timeout_ms

        self._pending: list[PendingRequest] = []
        self._lock = asyncio.Lock()
        self._timeout_task: Optional[asyncio.Task] = None
        self._running = False

        # Aggregate metrics across all batches
        self._total_batches = 0
        self._total_requests = 0
        self._batch_sizes: list[int] = []
        self._queue_wait_times: list[float] = []
        self._inference_times: list[float] = []

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background timeout processor."""
        self._running = True
        self._timeout_task = asyncio.create_task(
            self._timeout_processor(), name="batcher-timeout"
        )
        logger.info(
            "DynamicBatcher started (max_batch=%d, timeout=%.0fms)",
            self.max_batch_size, self.batch_timeout_ms
        )

    async def stop(self) -> None:
        """Gracefully cancel the background processor."""
        self._running = False
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
        logger.info("DynamicBatcher stopped.")

    # ── Public Submit API ─────────────────────────────────────────────────────

    async def submit(self, prompt: str, max_new_tokens: int) -> str:
        """
        Enqueue a request and wait for its result.

        Returns the generated text string.
        Raises any exception produced by the inference function.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        request = PendingRequest(prompt=prompt, max_new_tokens=max_new_tokens,
                                  future=future)

        async with self._lock:
            self._pending.append(request)
            should_process = len(self._pending) >= self.max_batch_size

        if should_process:
            # Batch is full — dispatch immediately (don't await, let it run)
            asyncio.create_task(self._process_batch(trigger="size"))

        return await future

    # ── Internal Batch Processing ─────────────────────────────────────────────

    async def _timeout_processor(self) -> None:
        """
        Background loop: fires every batch_timeout_ms and processes whatever
        is waiting, even if the batch isn't full yet.
        """
        interval = self.batch_timeout_ms / 1000.0
        while self._running:
            await asyncio.sleep(interval)
            async with self._lock:
                has_pending = len(self._pending) > 0
            if has_pending:
                await self._process_batch(trigger="timeout")

    async def _process_batch(self, trigger: str) -> None:
        """
        Drain the pending queue (up to max_batch_size), run inference,
        and resolve each request's future.
        """
        async with self._lock:
            if not self._pending:
                return
            # Take up to max_batch_size requests
            batch = self._pending[: self.max_batch_size]
            self._pending = self._pending[self.max_batch_size :]

        if not batch:
            return

        batch_start = time.monotonic()
        queue_wait_ms = (batch_start - batch[0].enqueued_at) * 1000

        prompts = [r.prompt for r in batch]
        max_new_tokens = max(r.max_new_tokens for r in batch)

        logger.debug(
            "Processing batch: size=%d trigger=%s queue_wait=%.1fms",
            len(batch), trigger, queue_wait_ms
        )

        try:
            inference_start = time.monotonic()
            results = await self._inference_fn(prompts, max_new_tokens)
            inference_ms = (time.monotonic() - inference_start) * 1000
        except Exception as exc:
            # Propagate exception to every waiting request
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)
            logger.error("Batch inference failed: %s", exc)
            return

        total_ms = (time.monotonic() - batch_start) * 1000

        # Resolve futures
        for req, result in zip(batch, results):
            if not req.future.done():
                req.future.set_result(result)

        # Record metrics
        self._total_batches += 1
        self._total_requests += len(batch)
        self._batch_sizes.append(len(batch))
        self._queue_wait_times.append(queue_wait_ms)
        self._inference_times.append(inference_ms)

        logger.info(
            "Batch done: size=%d trigger=%s inference=%.0fms total=%.0fms",
            len(batch), trigger, inference_ms, total_ms
        )

    # ── Metrics ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return aggregate batcher statistics."""
        n = len(self._batch_sizes)
        return {
            "total_batches": self._total_batches,
            "total_requests": self._total_requests,
            "avg_batch_size": round(sum(self._batch_sizes) / n, 2) if n else 0,
            "max_batch_size_seen": max(self._batch_sizes) if n else 0,
            "avg_queue_wait_ms": round(sum(self._queue_wait_times) / n, 2) if n else 0,
            "avg_inference_ms": round(sum(self._inference_times) / n, 2) if n else 0,
            "pending_requests": len(self._pending),
            "config_max_batch_size": self.max_batch_size,
            "config_timeout_ms": self.batch_timeout_ms,
        }

    def reset_stats(self) -> None:
        """Reset all aggregate counters (useful between benchmark runs)."""
        self._total_batches = 0
        self._total_requests = 0
        self._batch_sizes.clear()
        self._queue_wait_times.clear()
        self._inference_times.clear()
