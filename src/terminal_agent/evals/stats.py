"""Small statistics helpers (no numpy): Wilson intervals and a seeded bootstrap."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def rate(k: int, n: int) -> dict[str, object]:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": list(wilson(k, n))}


def cluster_rate(groups: Sequence[tuple[int, int]], iters: int = 2000, seed: int = 0
                 ) -> dict[str, object]:
    """Pooled rate over clusters (k_i of n_i), with a CI from resampling whole clusters.

    Hunks from one patch are not independent, so the interval resamples tasks, not hunks.
    """
    k = sum(g[0] for g in groups)
    n = sum(g[1] for g in groups)
    if not n:
        return {"k": 0, "n": 0, "clusters": 0, "rate": None, "ci95_cluster": [0.0, 0.0]}
    rng = random.Random(seed)
    draws = []
    for _ in range(iters):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        sn = sum(g[1] for g in sample)
        draws.append(sum(g[0] for g in sample) / sn if sn else 0.0)
    draws.sort()
    return {"k": k, "n": n, "clusters": len(groups), "rate": round(k / n, 4),
            "ci95_cluster": [round(draws[int(0.025 * iters)], 4),
                             round(draws[int(0.975 * iters) - 1], 4)]}


def bootstrap_mean_ci(values: Sequence[float], iters: int = 2000, seed: int = 0
                      ) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(iters))
    return (round(means[int(0.025 * iters)], 4), round(means[int(0.975 * iters) - 1], 4))
