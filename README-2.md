# LLM Inference Server

High-throughput LLM inference API with dynamic batching and in-process caching.

---

## Features

- **Dynamic batching** — groups concurrent requests into batches using a hybrid size/timeout trigger, maximising GPU/CPU utilisation
- **In-process response cache** — LRU cache with configurable TTL and max-entry limit; eliminates redundant inference for repeated prompts
- **Privacy-preserving cache keys** — prompts are SHA-256 hashed; no plaintext content or user identifiers are ever stored
- **Fully async** — FastAPI + asyncio handles high concurrency without blocking
- **Configurable** — all tuning parameters exposed via environment variables
- **Benchmark suite** — comprehensive latency, throughput, and cache-effectiveness tests

---

## Quick Start

### Prerequisites

- Python 3.10+
- ~2 GB disk space (model weights downloaded on first run)
- GPU recommended but not required (CPU mode works with the default tiny model)

### Installation

```bash
git clone <your-repo-url>
cd ids568-milestone5-<netid>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env to change model, batch size, cache TTL, etc.
```

---

## Running the Server

### Development mode (auto-reload)

```bash
uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
```

### Production mode

```bash
python -m src.server
```

Or with explicit settings:

```bash
LLM_MAX_BATCH_SIZE=16 LLM_BATCH_TIMEOUT_MS=30 python -m src.server
```

---

## API Usage

### Generate text

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain machine learning", "max_new_tokens": 50}'
```

Response:
```json
{
  "generated_text": "...",
  "cached": false,
  "latency_ms": 312.4,
  "model": "sshleifer/tiny-gpt2"
}
```

### Health check

```bash
curl http://localhost:8000/health
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

Returns batching stats, cache hit rate, model info, and config snapshot.

### Clear cache (admin)

```bash
curl -X POST http://localhost:8000/cache/clear
```

---

## Running Benchmarks

Make sure the server is running first, then:

```bash
# Full benchmark suite (recommended)
python benchmarks/run_benchmarks.py

# Quick run with fewer requests
python benchmarks/run_benchmarks.py --quick --requests 20

# Against a remote server
python benchmarks/run_benchmarks.py --url http://<host>:8000

# See all options
python benchmarks/run_benchmarks.py --help
```

Results are saved to `benchmarks/results/benchmark_results.json`.

### Benchmark tests included

| Test | What it measures |
|------|-----------------|
| Single-request baseline | Latency without batching benefit |
| Batched concurrent | Throughput gain from batching |
| Cold vs warm cache | Cache latency reduction |
| Cache hit rate over time | Hit rate evolution as prompts repeat |
| Throughput levels | RPS at concurrency 1 / 4 / 8 / 16 |
| Batch size comparison | Effect of concurrency on batch formation |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
ids568-milestone5-<netid>/
├── src/
│   ├── server.py       # FastAPI app, endpoints, lifespan
│   ├── batching.py     # DynamicBatcher — hybrid size/timeout batching
│   ├── caching.py      # InProcessCache — LRU + TTL, privacy-safe keys
│   ├── inference.py    # HuggingFace model loading and batch inference
│   └── config.py       # Pydantic settings (LLM_ env prefix)
├── benchmarks/
│   ├── run_benchmarks.py   # Benchmark orchestration (run this)
│   ├── load_generator.py   # Synthetic prompt generator + async HTTP client
│   └── results/            # JSON results written here after benchmarks
├── analysis/
│   ├── performance_report.pdf   # Generated after benchmark run
│   ├── governance_memo.pdf      # Privacy and retention analysis
│   └── visualizations/          # Charts produced from benchmark data
├── .env.example        # Copy to .env and customise
├── requirements.txt
└── README.md
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_NAME` | `sshleifer/tiny-gpt2` | HuggingFace model ID |
| `LLM_MAX_NEW_TOKENS` | `50` | Max tokens generated per request |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature (0 = greedy) |
| `LLM_MAX_BATCH_SIZE` | `8` | Max requests per batch |
| `LLM_BATCH_TIMEOUT_MS` | `50` | Batch timeout in milliseconds |
| `LLM_MAX_SEQUENCE_LENGTH` | `512` | Max tokens in prompt (truncated if exceeded) |
| `LLM_CACHE_ENABLED` | `true` | Enable/disable caching |
| `LLM_CACHE_TTL_SECONDS` | `3600` | Cache entry time-to-live |
| `LLM_CACHE_MAX_ENTRIES` | `10000` | LRU eviction threshold |
| `LLM_HOST` | `0.0.0.0` | Bind host |
| `LLM_PORT` | `8000` | Bind port |

---

## License

MIT
