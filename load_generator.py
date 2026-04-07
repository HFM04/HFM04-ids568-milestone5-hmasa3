"""
Synthetic load generator for LLM inference benchmarking.

Generates realistic prompts with configurable:
  - Mix of unique vs repeated prompts (to test cache hit rate)
  - Prompt length variation
  - Concurrency levels
"""

import random
import asyncio
import time
import httpx
from dataclasses import dataclass, field
from typing import Optional

# ── Prompt pool ───────────────────────────────────────────────────────────────

UNIQUE_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "What is the difference between supervised and unsupervised learning?",
    "Describe how neural networks work.",
    "What are the key principles of data engineering?",
    "How does gradient descent optimize a model?",
    "What is overfitting and how do you prevent it?",
    "Explain the transformer architecture.",
    "What is transfer learning?",
    "How does backpropagation work?",
    "What is the role of activation functions in neural networks?",
    "Explain batch normalization.",
    "What is dropout regularization?",
    "How does attention mechanism work in NLP?",
    "What is the difference between BERT and GPT?",
    "Explain the concept of embeddings.",
    "What is a convolutional neural network?",
    "How do recurrent neural networks handle sequences?",
    "What is reinforcement learning?",
    "Explain the bias-variance tradeoff.",
    "What is cross-validation?",
    "How does random forest work?",
    "What is gradient boosting?",
    "Explain support vector machines.",
    "What is dimensionality reduction?",
    "How does PCA work?",
    "What is k-means clustering?",
    "Explain the concept of precision and recall.",
    "What is the ROC curve?",
    "How do you handle imbalanced datasets?",
    "What is feature engineering?",
]

REPEATED_PROMPTS = [
    "What is machine learning?",
    "Explain deep learning.",
    "What is artificial intelligence?",
    "How does Python work?",
    "What is a neural network?",
]


@dataclass
class RequestResult:
    """Result of a single benchmark request."""
    prompt: str
    success: bool
    latency_ms: float
    cached: bool
    generated_text: str = ""
    error: str = ""
    timestamp: float = field(default_factory=time.monotonic)


class LoadGenerator:
    """
    Async load generator that sends concurrent requests to the inference server.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        max_new_tokens: int = 50,
        timeout_s: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout_s = timeout_s

    def build_prompt_list(
        self,
        n_requests: int,
        repeat_ratio: float = 0.3,
        seed: int = 42,
    ) -> list[str]:
        """
        Build a list of prompts with a controlled repeat ratio.

        Args:
            n_requests: Total number of prompts to generate.
            repeat_ratio: Fraction of prompts drawn from the repeated set (0–1).
                          Higher = better cache hit rate.
            seed: Random seed for reproducibility.

        Returns:
            List of prompt strings.
        """
        rng = random.Random(seed)
        prompts = []
        for _ in range(n_requests):
            if rng.random() < repeat_ratio:
                prompts.append(rng.choice(REPEATED_PROMPTS))
            else:
                prompts.append(rng.choice(UNIQUE_PROMPTS))
        return prompts

    async def _send_request(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        semaphore: asyncio.Semaphore,
        use_cache: bool = True,
    ) -> RequestResult:
        """Send a single /generate request and record the result."""
        async with semaphore:
            t_start = time.monotonic()
            try:
                resp = await client.post(
                    f"{self.base_url}/generate",
                    json={
                        "prompt": prompt,
                        "max_new_tokens": self.max_new_tokens,
                        "use_cache": use_cache,
                    },
                    timeout=self.timeout_s,
                )
                latency_ms = (time.monotonic() - t_start) * 1000
                if resp.status_code == 200:
                    data = resp.json()
                    return RequestResult(
                        prompt=prompt,
                        success=True,
                        latency_ms=round(latency_ms, 2),
                        cached=data.get("cached", False),
                        generated_text=data.get("generated_text", ""),
                    )
                else:
                    return RequestResult(
                        prompt=prompt,
                        success=False,
                        latency_ms=round(latency_ms, 2),
                        cached=False,
                        error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                    )
            except Exception as exc:
                latency_ms = (time.monotonic() - t_start) * 1000
                return RequestResult(
                    prompt=prompt,
                    success=False,
                    latency_ms=round(latency_ms, 2),
                    cached=False,
                    error=str(exc),
                )

    async def run(
        self,
        prompts: list[str],
        concurrency: int = 10,
        use_cache: bool = True,
    ) -> list[RequestResult]:
        """
        Send all prompts concurrently (up to `concurrency` at a time).

        Args:
            prompts: List of prompt strings to send.
            concurrency: Maximum simultaneous in-flight requests.
            use_cache: Whether to allow cache hits.

        Returns:
            List of RequestResult in completion order.
        """
        semaphore = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient() as client:
            tasks = [
                self._send_request(client, p, semaphore, use_cache)
                for p in prompts
            ]
            results = await asyncio.gather(*tasks)
        return list(results)

    async def warmup(self, n: int = 3) -> None:
        """Send a few requests to warm up the model and JIT caches."""
        print(f"  Warming up with {n} requests...")
        prompts = REPEATED_PROMPTS[:n]
        await self.run(prompts, concurrency=1, use_cache=False)
        print("  Warmup complete.")
