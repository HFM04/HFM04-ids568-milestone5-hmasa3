"""
Model loading and inference execution.

Uses HuggingFace Transformers with a CPU-friendly model by default.
The inference function is async-wrapped so it integrates cleanly with
the asyncio event loop without blocking it (runs in a thread executor).
"""

import asyncio
import logging
import time
from functools import partial
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from src.config import settings

logger = logging.getLogger(__name__)

# ── Global model/tokenizer references ────────────────────────────────────────
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None
_pipeline = None
_model_load_time_s: float = 0.0


def load_model() -> None:
    """
    Load tokenizer and model into memory (blocking).
    Call once at server startup inside the lifespan context.
    """
    global _tokenizer, _model, _pipeline, _model_load_time_s

    logger.info("Loading model: %s", settings.model_name)
    t0 = time.monotonic()

    _tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

    # Ensure pad token exists (GPT-2 family doesn't have one by default)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        torch_dtype=torch.float32,   # CPU-safe precision
        low_cpu_mem_usage=True,
    )
    _model.eval()

    # Use a pipeline for convenient batched generation
    _pipeline = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
        device=-1,            # -1 = CPU
        batch_size=settings.max_batch_size,
    )

    _model_load_time_s = time.monotonic() - t0
    logger.info("Model loaded in %.2fs", _model_load_time_s)


def _run_inference_sync(prompts: list[str], max_new_tokens: int) -> list[str]:
    """
    Synchronous batch inference. Called in a thread executor to avoid
    blocking the asyncio event loop.
    """
    if _pipeline is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Truncate prompts that exceed max sequence length
    truncated = []
    for p in prompts:
        tokens = _tokenizer.encode(p)
        if len(tokens) > settings.max_sequence_length - max_new_tokens:
            tokens = tokens[: settings.max_sequence_length - max_new_tokens]
            p = _tokenizer.decode(tokens, skip_special_tokens=True)
        truncated.append(p)

    outputs = _pipeline(
        truncated,
        max_new_tokens=max_new_tokens,
        do_sample=settings.temperature > 0,
        temperature=settings.temperature if settings.temperature > 0 else None,
        pad_token_id=_tokenizer.pad_token_id,
        return_full_text=False,   # Return only the generated continuation
    )

    # pipeline returns list[list[dict]] for batched input
    results = []
    for output in outputs:
        if isinstance(output, list):
            results.append(output[0]["generated_text"])
        else:
            results.append(output["generated_text"])
    return results


async def run_inference(prompts: list[str], max_new_tokens: int) -> list[str]:
    """
    Async wrapper: executes synchronous inference in a thread pool so the
    event loop remains free to accept new requests during GPU/CPU work.
    """
    loop = asyncio.get_event_loop()
    fn = partial(_run_inference_sync, prompts, max_new_tokens)
    return await loop.run_in_executor(None, fn)


def model_info() -> dict:
    """Return metadata about the loaded model."""
    if _model is None:
        return {"status": "not_loaded"}
    param_count = sum(p.numel() for p in _model.parameters())
    return {
        "model_name": settings.model_name,
        "parameters": param_count,
        "parameters_millions": round(param_count / 1e6, 1),
        "dtype": str(next(_model.parameters()).dtype),
        "device": "cpu",
        "load_time_s": round(_model_load_time_s, 2),
    }
