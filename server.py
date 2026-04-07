"""
LLM Inference Server — FastAPI entry point.

Endpoints:
  POST /generate   - Submit a prompt, get generated text
  GET  /health     - Liveness check
  GET  /metrics    - Server + batching + cache statistics
  POST /cache/clear - Manually flush the cache (admin utility)

Architecture:
  Request → Cache lookup → (hit) return immediately
                         → (miss) DynamicBatcher.submit()
                                  → batch full OR timeout fires
                                  → inference_fn(batch)
                                  → cache store + return
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.caching import cache, make_cache_key, should_cache
from src.batching import DynamicBatcher
from src.inference import load_model, run_inference, model_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Batcher (initialised in lifespan) ────────────────────────────────────────
batcher: DynamicBatcher = None  # type: ignore


# ── Lifespan: startup / shutdown ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global batcher

    # ── Startup ──
    logger.info("=== Server starting up ===")
    load_model()
    batcher = DynamicBatcher(
        inference_fn=run_inference,
        max_batch_size=settings.max_batch_size,
        batch_timeout_ms=settings.batch_timeout_ms,
    )
    await batcher.start()
    logger.info("=== Server ready ===")

    yield  # Server is live here

    # ── Shutdown ──
    logger.info("=== Server shutting down ===")
    await batcher.stop()
    logger.info("=== Shutdown complete ===")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLM Inference Server",
    description="High-throughput LLM API with dynamic batching and in-process caching.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096,
                        description="Input text prompt.")
    max_new_tokens: int = Field(
        default=None, ge=1, le=512,
        description="Max tokens to generate. Defaults to server config."
    )
    temperature: float = Field(
        default=None, ge=0.0, le=2.0,
        description="Sampling temperature. Defaults to server config (0.0 = greedy)."
    )
    use_cache: bool = Field(
        default=True,
        description="Set False to bypass cache and always run inference."
    )


class GenerateResponse(BaseModel):
    generated_text: str
    cached: bool
    latency_ms: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_s: float


# ── Server start time for uptime tracking ─────────────────────────────────────
_start_time = time.monotonic()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """
    Generate text for the given prompt.

    - Checks cache first (if temperature == 0.0 and use_cache == True).
    - On cache miss, submits to the dynamic batcher and waits for result.
    - Stores result in cache on miss.
    """
    t_start = time.monotonic()

    temperature = req.temperature if req.temperature is not None else settings.temperature
    max_new_tokens = req.max_new_tokens if req.max_new_tokens is not None else settings.max_new_tokens

    # ── Cache lookup ──────────────────────────────────────────────────────
    cached_result = None
    cache_key = None

    if settings.cache_enabled and req.use_cache and should_cache(temperature):
        cache_key = make_cache_key(
            prompt=req.prompt,
            model=settings.model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        cached_result = await cache.get(cache_key)

    if cached_result is not None:
        latency_ms = (time.monotonic() - t_start) * 1000
        return GenerateResponse(
            generated_text=cached_result,
            cached=True,
            latency_ms=round(latency_ms, 2),
            model=settings.model_name,
        )

    # ── Batch inference ───────────────────────────────────────────────────
    try:
        result = await batcher.submit(req.prompt, max_new_tokens)
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    # ── Cache store ───────────────────────────────────────────────────────
    if cache_key is not None:
        await cache.set(cache_key, result)

    latency_ms = (time.monotonic() - t_start) * 1000
    return GenerateResponse(
        generated_text=result,
        cached=False,
        latency_ms=round(latency_ms, 2),
        model=settings.model_name,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe — always returns 200 if the server is up."""
    return HealthResponse(
        status="ok",
        model_loaded=True,
        uptime_s=round(time.monotonic() - _start_time, 1),
    )


@app.get("/metrics")
async def metrics():
    """
    Aggregate server metrics including:
    - Model info
    - Batching statistics
    - Cache statistics
    - Configuration snapshot
    """
    return JSONResponse({
        "model": model_info(),
        "batching": batcher.stats() if batcher else {},
        "cache": await cache.stats(),
        "config": {
            "max_batch_size": settings.max_batch_size,
            "batch_timeout_ms": settings.batch_timeout_ms,
            "cache_ttl_seconds": settings.cache_ttl_seconds,
            "cache_max_entries": settings.cache_max_entries,
            "cache_enabled": settings.cache_enabled,
            "max_new_tokens": settings.max_new_tokens,
            "temperature": settings.temperature,
        },
    })


@app.post("/cache/clear")
async def clear_cache():
    """Admin endpoint: flush all cache entries."""
    count = await cache.clear()
    batcher.reset_stats() if batcher else None
    return {"cleared_entries": count, "message": "Cache and batcher stats reset."}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "src.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )
