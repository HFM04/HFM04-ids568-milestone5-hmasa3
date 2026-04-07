#!/usr/bin/env python3
"""
Benchmark orchestration for Milestone 5.

Runs a comprehensive suite of performance tests and saves results to
benchmarks/results/ as JSON files ready for analysis and visualization.

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --url http://localhost:8000
    python benchmarks/run_benchmarks.py --skip-warmup --requests 50
    python benchmarks/run_benchmarks.py --help
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from src.load_generator import LoadGenerator, RequestResult


RESULTS_DIR = Path(__file__).parent / "results"


# ── Utilities ─────────────────────────────────────────────────────────────────

def summarise(results: list[RequestResult], label: str) -> dict:
    """Compute summary statistics from a list of RequestResult."""
    successful = [r for r in results if r.success]
    latencies = [r.latency_ms for r in successful]
    cached_count = sum(1 for r in successful if r.cached)

    if not latencies:
        return {"label": label, "error": "All requests failed"}

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    def percentile(p):
        idx = max(0, int(p / 100 * n) - 1)
        return round(latencies_sorted[idx], 2)

    return {
        "label": label,
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "cached": cached_count,
        "cache_hit_rate": round(cached_count / len(successful), 4) if successful else 0,
        "latency_ms": {
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
            "mean": round(statistics.mean(latencies), 2),
            "median": round(statistics.median(latencies), 2),
            "p50": percentile(50),
            "p75": percentile(75),
            "p95": percentile(95),
            "p99": percentile(99),
            "stdev": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
        },
    }


def save_results(data: dict, filename: str) -> Path:
    """Save benchmark results as JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved -> {path}")
    return path


async def clear_cache(base_url: str) -> None:
    """POST /cache/clear to reset state between tests."""
    async with httpx.AsyncClient() as client:
        await client.post(f"{base_url}/cache/clear", timeout=10)


async def get_metrics(base_url: str) -> dict:
    """Fetch /metrics from the server."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{base_url}/metrics", timeout=10)
        return resp.json()


def print_summary(summary: dict) -> None:
    """Pretty-print a summary dict to stdout."""
    lat = summary.get("latency_ms", {})
    print(f"  Requests : {summary['total_requests']} "
          f"(✓ {summary['successful']}  ✗ {summary['failed']})")
    print(f"  Cache    : {summary['cached']} hits "
          f"({summary['cache_hit_rate']*100:.1f}%)")
    print(f"  Latency  : mean={lat.get('mean')}ms  "
          f"p50={lat.get('p50')}ms  "
          f"p95={lat.get('p95')}ms  "
          f"p99={lat.get('p99')}ms")


# ── Individual Benchmark Tests ────────────────────────────────────────────────

async def test_single_request(gen: LoadGenerator, base_url: str) -> dict:
    """Baseline: single sequential requests, no batching benefit."""
    print("\n[1] Single-request baseline (sequential, no cache) ...")
    await clear_cache(base_url)

    prompts = gen.build_prompt_list(n_requests=20, repeat_ratio=0.0, seed=1)
    t0 = time.monotonic()
    results = await gen.run(prompts, concurrency=1, use_cache=False)
    elapsed = time.monotonic() - t0

    summary = summarise(results, "single_request_baseline")
    summary["throughput_rps"] = round(len(results) / elapsed, 2)
    summary["elapsed_s"] = round(elapsed, 2)
    print_summary(summary)
    return summary


async def test_batched_requests(gen: LoadGenerator, base_url: str,
                                 concurrency: int, n_requests: int,
                                 label: str) -> dict:
    """Concurrent requests that trigger the dynamic batcher."""
    print(f"\n[2] Batched requests — concurrency={concurrency} ...")
    await clear_cache(base_url)

    prompts = gen.build_prompt_list(n_requests=n_requests, repeat_ratio=0.0, seed=2)
    t0 = time.monotonic()
    results = await gen.run(prompts, concurrency=concurrency, use_cache=False)
    elapsed = time.monotonic() - t0

    summary = summarise(results, label)
    summary["concurrency"] = concurrency
    summary["throughput_rps"] = round(len(results) / elapsed, 2)
    summary["elapsed_s"] = round(elapsed, 2)
    print_summary(summary)
    return summary


async def test_cache_cold_vs_warm(gen: LoadGenerator, base_url: str,
                                   n_requests: int) -> dict:
    """Compare cold-cache vs warm-cache latency."""
    print("\n[3] Cold vs warm cache comparison ...")

    # Cold pass (cache empty)
    await clear_cache(base_url)
    prompts = gen.build_prompt_list(n_requests=n_requests, repeat_ratio=1.0, seed=3)
    results_cold = await gen.run(prompts, concurrency=5, use_cache=True)
    cold_summary = summarise(results_cold, "cold_cache")
    cold_summary["pass"] = "cold"

    # Warm pass (same prompts — should be cache hits)
    results_warm = await gen.run(prompts, concurrency=5, use_cache=True)
    warm_summary = summarise(results_warm, "warm_cache")
    warm_summary["pass"] = "warm"

    print("  Cold cache:")
    print_summary(cold_summary)
    print("  Warm cache:")
    print_summary(warm_summary)

    speedup = (cold_summary["latency_ms"]["mean"] /
               warm_summary["latency_ms"]["mean"]
               if warm_summary["latency_ms"]["mean"] > 0 else 0)
    print(f"  Cache speedup: {speedup:.1f}x")

    return {
        "cold": cold_summary,
        "warm": warm_summary,
        "speedup_x": round(speedup, 2),
    }


async def test_cache_hit_rate_over_time(gen: LoadGenerator, base_url: str,
                                         n_requests: int,
                                         repeat_ratio: float) -> dict:
    """Track how cache hit rate evolves as repeated prompts accumulate."""
    print(f"\n[4] Cache hit-rate over time (repeat_ratio={repeat_ratio}) ...")
    await clear_cache(base_url)

    prompts = gen.build_prompt_list(n_requests=n_requests,
                                    repeat_ratio=repeat_ratio, seed=4)
    results = await gen.run(prompts, concurrency=8, use_cache=True)

    # Build cumulative hit-rate series (window of 10)
    window = 10
    hit_rate_series = []
    for i in range(window, len(results) + 1, window):
        window_results = results[i - window: i]
        hits = sum(1 for r in window_results if r.success and r.cached)
        hit_rate_series.append({
            "request_index": i,
            "window_hit_rate": round(hits / window, 4),
        })

    summary = summarise(results, f"cache_hit_rate_repeat{repeat_ratio}")
    summary["repeat_ratio"] = repeat_ratio
    summary["hit_rate_series"] = hit_rate_series
    print_summary(summary)
    return summary


async def test_throughput_levels(gen: LoadGenerator, base_url: str,
                                  levels: list[int], n_requests: int) -> list[dict]:
    """Measure throughput at multiple load levels."""
    print("\n[5] Throughput at multiple load levels ...")
    results_all = []

    for concurrency in levels:
        await clear_cache(base_url)
        prompts = gen.build_prompt_list(n_requests=n_requests,
                                        repeat_ratio=0.0, seed=5 + concurrency)
        t0 = time.monotonic()
        results = await gen.run(prompts, concurrency=concurrency, use_cache=False)
        elapsed = time.monotonic() - t0

        summary = summarise(results, f"throughput_concurrency_{concurrency}")
        summary["concurrency"] = concurrency
        summary["throughput_rps"] = round(len([r for r in results if r.success]) / elapsed, 2)
        summary["elapsed_s"] = round(elapsed, 2)
        print(f"  concurrency={concurrency:3d}: "
              f"rps={summary['throughput_rps']:6.2f}  "
              f"p95={summary['latency_ms'].get('p95')}ms")
        results_all.append(summary)

    return results_all


async def test_batch_size_comparison(gen: LoadGenerator, base_url: str,
                                      n_requests: int) -> dict:
    """
    Compare effective batch behaviour by varying concurrency.
    The batcher groups concurrent requests — high concurrency → larger batches.
    """
    print("\n[6] Batch size comparison (low vs high concurrency) ...")
    comparisons = {}

    for label, concurrency in [("low_concurrency_1", 1),
                                 ("medium_concurrency_4", 4),
                                 ("high_concurrency_8", 8)]:
        await clear_cache(base_url)
        prompts = gen.build_prompt_list(n_requests=n_requests,
                                        repeat_ratio=0.0, seed=10)
        t0 = time.monotonic()
        results = await gen.run(prompts, concurrency=concurrency, use_cache=False)
        elapsed = time.monotonic() - t0

        metrics = await get_metrics(base_url)
        summary = summarise(results, label)
        summary["concurrency"] = concurrency
        summary["throughput_rps"] = round(len([r for r in results if r.success]) / elapsed, 2)
        summary["batcher_stats"] = metrics.get("batching", {})
        comparisons[label] = summary
        print(f"  {label}: rps={summary['throughput_rps']}  "
              f"p95={summary['latency_ms'].get('p95')}ms  "
              f"avg_batch={metrics.get('batching', {}).get('avg_batch_size', '?')}")

    return comparisons


# ── Main Orchestration ────────────────────────────────────────────────────────

async def run_all(args) -> None:
    base_url = args.url
    n = args.requests
    gen = LoadGenerator(base_url=base_url, max_new_tokens=args.max_new_tokens)

    print(f"\n{'='*60}")
    print(f"  Milestone 5 Benchmark Suite")
    print(f"  Server : {base_url}")
    print(f"  Requests per test : {n}")
    print(f"{'='*60}")

    # Warmup
    if not args.skip_warmup:
        print("\n[0] Warming up ...")
        await gen.warmup(n=3)

    all_results = {"run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "config": vars(args)}

    # 1. Single request baseline
    all_results["single_request"] = await test_single_request(gen, base_url)

    # 2. Batched requests
    all_results["batched_concurrent"] = await test_batched_requests(
        gen, base_url, concurrency=8, n_requests=n, label="batched_concurrent_8"
    )

    # 3. Cold vs warm cache
    all_results["cache_cold_vs_warm"] = await test_cache_cold_vs_warm(
        gen, base_url, n_requests=min(n, 30)
    )

    # 4. Cache hit rate over time
    all_results["cache_hit_rate_30pct"] = await test_cache_hit_rate_over_time(
        gen, base_url, n_requests=n, repeat_ratio=0.3
    )
    all_results["cache_hit_rate_60pct"] = await test_cache_hit_rate_over_time(
        gen, base_url, n_requests=n, repeat_ratio=0.6
    )

    # 5. Throughput at multiple load levels
    load_levels = [1, 4, 8, 16] if not args.quick else [1, 4]
    all_results["throughput_levels"] = await test_throughput_levels(
        gen, base_url, levels=load_levels, n_requests=n
    )

    # 6. Batch size comparison
    all_results["batch_comparison"] = await test_batch_size_comparison(
        gen, base_url, n_requests=min(n, 24)
    )

    # Final server metrics snapshot
    all_results["final_server_metrics"] = await get_metrics(base_url)

    # Save all results
    print("\n" + "="*60)
    print("Saving results ...")
    save_results(all_results, "benchmark_results.json")

    # Also save individual files for easier analysis
    for key in ["single_request", "batched_concurrent",
                "cache_cold_vs_warm", "throughput_levels"]:
        if key in all_results:
            save_results(all_results[key], f"{key}.json")

    print("\n✓ All benchmarks complete.")
    print(f"  Results saved to: {RESULTS_DIR}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Milestone 5 Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the inference server.")
    parser.add_argument("--requests", type=int, default=50,
                        help="Number of requests per test.")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Max tokens to generate per request.")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip the warmup phase.")
    parser.add_argument("--quick", action="store_true",
                        help="Run a quick subset of tests.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_all(args))
