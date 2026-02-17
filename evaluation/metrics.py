"""Metrics collection for Melting Pot evaluation.

Tracks focal per-capita return (the official primary metric) and
secondary metrics per episode and across episodes.
"""
from __future__ import annotations

import math
from typing import Any


class EpisodeMetrics:
    """Collects metrics for a single episode."""

    def __init__(self, n_focal: int) -> None:
        self.n_focal = n_focal
        self.focal_rewards: list[float] = []
        self.steps: int = 0

    def record_step(self, focal_rewards: list[float]) -> None:
        """Record one step's focal agent rewards."""
        self.steps += 1
        self.focal_rewards.extend(focal_rewards)

    def compute(self) -> dict[str, Any]:
        """Compute final metrics for the episode."""
        n = max(self.n_focal, 1)
        total_focal = sum(self.focal_rewards)
        per_capita = total_focal / n if self.steps > 0 else 0.0

        return {
            "focal_per_capita": per_capita,
            "focal_total": total_focal,
            "n_focal": self.n_focal,
            "steps": self.steps,
        }


def aggregate_episode_metrics(
    episode_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate metrics across episodes.

    Returns mean, std, 95% CI, min, max for focal per-capita return.
    """
    if not episode_metrics:
        return {}

    per_capitas = [ep["focal_per_capita"] for ep in episode_metrics]
    n = len(per_capitas)
    mean = sum(per_capitas) / n

    if n > 1:
        variance = sum((v - mean) ** 2 for v in per_capitas) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    ci95 = 2.0 * std / math.sqrt(n) if n > 0 else 0.0

    return {
        "n_episodes": n,
        "focal_per_capita": {
            "mean": mean,
            "std": std,
            "ci95": ci95,
            "min": min(per_capitas),
            "max": max(per_capitas),
            "all": per_capitas,
        },
        "mean_steps": sum(ep["steps"] for ep in episode_metrics) / n,
    }
