"""Cross-platform HTTP load probe using only the Python standard library."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


def request(url: str) -> tuple[float, int]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/ready", timeout=10) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
    except (OSError, TimeoutError):
        status = 0
    return (time.perf_counter() - started) * 1000, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda _: request(args.url), range(args.requests)))
    durations = [duration for duration, _ in results]
    statuses: dict[str, int] = {}
    for _, status in results:
        statuses[str(status)] = statuses.get(str(status), 0) + 1
    print(
        json.dumps(
            {
                "requests": args.requests,
                "workers": args.workers,
                "status_counts": statuses,
                "latency_ms": {
                    "p50": percentile(durations, 0.50),
                    "p95": percentile(durations, 0.95),
                    "p99": percentile(durations, 0.99),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
