# metrics.py -- Per-capita reward, cooperation events, resource efficiency.
# Normalised against DeepMind baselines (worst=0, best=1).
"""
Evaluation metrics collected per episode and aggregated across episodes.

Tracks: per-agent rewards, cooperation/defection events, resource harvested
and remaining, Gini coefficient of reward distribution, sustainability.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Cooperation heuristics by action index.
#
# Melting Pot discrete actions 0-7 vary by substrate, but a common mapping:
#   0 = noop, 1-4 = move, 5 = turn left, 6 = turn right, 7 = interact
# "interact" (7) is the cooperation/defection-relevant action in most
# substrates.  We flag it as cooperative when reward context is non-negative
# (a simplification -- substrate-specific detectors can refine this later).
# ---------------------------------------------------------------------------

INTERACT_ACTION = 7


def _gini(values: list[float]) -> float:
    """Compute the Gini coefficient for a list of values.

    0 = perfect equality, 1 = maximum inequality.
    Returns 0.0 for empty or all-zero inputs.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumulative = 0.0
    numerator = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += v
        numerator += (2 * (i + 1) - n - 1) * v
    return float(numerator / (n * total))


class MetricsCollector:
    """Collects and computes evaluation metrics for a single episode."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.per_agent_rewards: dict[str, float] = defaultdict(float)
        self.per_agent_actions: dict[str, list[int]] = defaultdict(list)
        self.cooperation_events: int = 0
        self.defection_events: int = 0
        self.resources_harvested: float = 0.0
        self.resources_remaining: float = 0.0
        self.steps: int = 0
        self._positive_reward_steps: int = 0
        self._total_reward: float = 0.0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        agent_id: str,
        reward: float,
        action: int,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Record one agent step within the episode."""
        self.steps += 1
        self.per_agent_rewards[agent_id] += reward
        self.per_agent_actions[agent_id].append(action)
        self._total_reward += reward

        if reward > 0:
            self._positive_reward_steps += 1

        # Cooperation heuristic: interact action with non-negative reward
        if action == INTERACT_ACTION:
            if reward >= 0:
                self.cooperation_events += 1
            else:
                self.defection_events += 1

        # Substrate-specific resource tracking from info dict
        if info:
            self.resources_harvested += info.get("resources_harvested", 0.0)
            self.resources_remaining = info.get("resources_remaining",
                                                self.resources_remaining)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(self) -> dict[str, Any]:
        """Compute final metrics for the episode.

        Returns a dict with all metrics.  Keys:
            total_reward, per_capita_reward, cooperation_ratio,
            cooperation_events, defection_events, resource_efficiency,
            gini_coefficient, sustainability, steps, n_agents,
            per_agent_rewards.
        """
        n = max(self.n_agents, 1)
        total = self._total_reward
        per_capita = total / n

        coop = self.cooperation_events
        defect = self.defection_events
        coop_total = coop + defect
        cooperation_ratio = coop / coop_total if coop_total > 0 else 0.0

        harvested = self.resources_harvested
        remaining = self.resources_remaining
        total_resource = harvested + remaining
        resource_efficiency = (
            harvested / total_resource if total_resource > 0 else 0.0
        )

        # Sustainability: fraction of episode with positive reward events
        sustainability = (
            self._positive_reward_steps / max(self.steps, 1)
        )

        reward_list = list(self.per_agent_rewards.values())
        # Shift rewards to non-negative for Gini (Gini is defined for >= 0)
        min_r = min(reward_list) if reward_list else 0.0
        if min_r < 0:
            shifted = [r - min_r for r in reward_list]
        else:
            shifted = reward_list
        gini = _gini(shifted)

        return {
            "total_reward": total,
            "per_capita_reward": per_capita,
            "cooperation_ratio": cooperation_ratio,
            "cooperation_events": coop,
            "defection_events": defect,
            "resource_efficiency": resource_efficiency,
            "gini_coefficient": gini,
            "sustainability": sustainability,
            "steps": self.steps,
            "n_agents": self.n_agents,
            "per_agent_rewards": dict(self.per_agent_rewards),
        }


# ---------------------------------------------------------------------------
# Aggregation across episodes
# ---------------------------------------------------------------------------

def aggregate_metrics(episode_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of per-episode metric dicts.

    Returns mean, std, min, max, and 95 % confidence interval half-width
    for the key scalar metrics.
    """
    if not episode_metrics:
        return {}

    scalar_keys = [
        "total_reward",
        "per_capita_reward",
        "cooperation_ratio",
        "cooperation_events",
        "defection_events",
        "resource_efficiency",
        "gini_coefficient",
        "sustainability",
        "steps",
    ]

    agg: dict[str, Any] = {"n_episodes": len(episode_metrics)}

    for key in scalar_keys:
        values = [ep[key] for ep in episode_metrics if key in ep]
        if not values:
            continue
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            variance = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        # 95 % CI (t ~ 1.96 for large n; use 2.0 as conservative approx)
        ci95 = 2.0 * std / math.sqrt(n) if n > 0 else 0.0
        agg[key] = {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
            "ci95": ci95,
        }

    return agg
