# Load test results (Locust)

Setup: API in Docker on an 8 GB / 4-thread Windows laptop (WSL2), Redis card state, champion v1.
Locust runs on the same laptop. Each request is a realistic `POST /score` (synthetic `LT` cards,
removed from Redis afterwards). 60 s per run, 0 failures in every valid run.

| Scenario | Throughput | p50 | p95 | p99 |
|---|---|---|---|---|
| 1 user, 1 worker (pure latency) | 12.8 req/s | 67 ms | 100 ms | 120 ms |
| 20 users, 1 worker | 31.8 req/s | 580 ms | 1,000 ms | 1,300 ms |
| 20 users, 3 workers (`API_WORKERS=3`) | 48.1 req/s | 320 ms | 970 ms | 1,500 ms |

## Findings
- **Saturation with 1 worker:** 20x the users gave only 2.5x the throughput but 9x the latency.
  Little's law holds (31.8 req/s x 0.58 s = 18.4 requests in flight for 20 users).
- **Bottleneck:** one Python process (GIL) at ~32 req/s, i.e. ~31 ms of CPU per request; profiling
  shows per-request pandas feature building dominates (model inference is smaller). Server-side
  latency is ~20 ms; the rest is the Windows -> WSL2 -> container path and Redis round trips.
- **3 workers:** +51% throughput and -45% p50, not 3x: the API workers, Locust, Redis and Docker
  share 4 CPU threads. p99 got worse (CPU contention). State in Redis made multi-worker scaling correct.
- **Targets not met on this laptop:** p99 < 50 ms and >= 500 req/s.

## Next steps
1. Run the API on dedicated hardware, separate from the load generator.
2. Scale out replicas (state is shared in Redis, so replicas scale nearly linearly).
3. Replace per-request pandas feature building with a lightweight path, guarded by the
   existing training/serving parity tests.
4. Measure a warmed-up p99 with a longer run (cold starts inflate short runs).
