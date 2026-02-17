# temporal.py -- Cooperation rate over time. Phase transitions. Convergence.
"""
Temporal dynamics analysis.

Analyses how metrics evolve within and across episodes:
    - Reward trajectory over time
    - Convergence detection (when does cooperation stabilise?)
    - Phase transition detection
    - Comparison of early vs late episode behaviour
"""
from __future__ import annotations

import math
from typing import Any


class TemporalAnalyzer:
    """Temporal dynamics analysis."""

    # ------------------------------------------------------------------
    # Reward trajectory
    # ------------------------------------------------------------------

    def reward_trajectory(
        self, data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Reward over time: does it improve?

        *data* is a list of window-level dicts (from
        ``EpisodeRecorder.get_reward_timeline()``), each containing
        ``mean_reward`` and ``window``.

        Returns:
            dict with slope, intercept, initial_reward, final_reward,
            improving (bool), n_windows.
        """
        if not data:
            return {
                "slope": 0.0,
                "intercept": 0.0,
                "initial_reward": 0.0,
                "final_reward": 0.0,
                "improving": False,
                "n_windows": 0,
            }

        xs = [float(d["window"]) for d in data]
        ys = [float(d["mean_reward"]) for d in data]
        slope, intercept = _linreg(xs, ys)

        return {
            "slope": slope,
            "intercept": intercept,
            "initial_reward": ys[0],
            "final_reward": ys[-1],
            "improving": slope > 0.0,
            "n_windows": len(data),
        }

    # ------------------------------------------------------------------
    # Convergence detection
    # ------------------------------------------------------------------

    def convergence_point(
        self,
        data: list[dict[str, Any]],
        metric_key: str = "cooperation_ratio",
        threshold: float = 0.02,
        min_stable_windows: int = 5,
    ) -> dict[str, Any]:
        """When does cooperation stabilise?

        Scans through window-level data and finds the first window after
        which the metric does not change by more than *threshold* for at
        least *min_stable_windows* consecutive windows.

        *data* is a list of window dicts containing *metric_key*.

        Returns:
            dict with converged (bool), convergence_window (int or None),
            convergence_value (float or None), total_windows (int).
        """
        values = [float(d.get(metric_key, 0.0)) for d in data]
        n = len(values)

        if n < min_stable_windows + 1:
            return {
                "converged": False,
                "convergence_window": None,
                "convergence_value": None,
                "total_windows": n,
            }

        for start in range(n - min_stable_windows):
            ref = values[start]
            stable = True
            for j in range(1, min_stable_windows + 1):
                if abs(values[start + j] - ref) > threshold:
                    stable = False
                    break
            if stable:
                return {
                    "converged": True,
                    "convergence_window": start,
                    "convergence_value": ref,
                    "total_windows": n,
                }

        return {
            "converged": False,
            "convergence_window": None,
            "convergence_value": None,
            "total_windows": n,
        }

    # ------------------------------------------------------------------
    # Phase transition detection
    # ------------------------------------------------------------------

    def detect_phase_transitions(
        self,
        data: list[dict[str, Any]],
        metric_key: str = "cooperation_ratio",
        sensitivity: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Detect sharp changes (phase transitions) in a metric timeline.

        A phase transition is flagged when the absolute change between
        consecutive windows exceeds *sensitivity* times the running
        standard deviation.

        Returns a list of transition dicts: {window, from_value, to_value,
        magnitude, z_score}.
        """
        values = [float(d.get(metric_key, 0.0)) for d in data]
        if len(values) < 3:
            return []

        deltas = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
        transitions: list[dict[str, Any]] = []

        running_mean = 0.0
        running_var = 0.0

        for i, delta in enumerate(deltas):
            if i == 0:
                running_mean = delta
                running_var = 0.0
                continue

            # Update running stats (Welford's online algorithm)
            old_mean = running_mean
            running_mean += (delta - running_mean) / (i + 1)
            running_var += (delta - old_mean) * (delta - running_mean)

            std = math.sqrt(running_var / i) if i > 0 else 1e-8
            if std < 1e-8:
                std = 1e-8

            z = (delta - running_mean) / std

            if z > sensitivity:
                transitions.append({
                    "window": i + 1,
                    "from_value": values[i],
                    "to_value": values[i + 1],
                    "magnitude": delta,
                    "z_score": z,
                })

        return transitions

    # ------------------------------------------------------------------
    # Early vs late comparison
    # ------------------------------------------------------------------

    def early_vs_late(
        self,
        data: list[dict[str, Any]],
        metric_key: str = "cooperation_ratio",
        split_fraction: float = 0.5,
    ) -> dict[str, Any]:
        """Compare the first half of an episode to the second half.

        Returns means and improvement ratio.
        """
        values = [float(d.get(metric_key, 0.0)) for d in data]
        if not values:
            return {
                "early_mean": 0.0,
                "late_mean": 0.0,
                "improvement": 0.0,
                "n_total": 0,
            }

        split = int(len(values) * split_fraction)
        early = values[:split] if split > 0 else values[:1]
        late = values[split:] if split < len(values) else values[-1:]

        early_mean = _mean(early)
        late_mean = _mean(late)
        improvement = late_mean - early_mean

        return {
            "early_mean": early_mean,
            "late_mean": late_mean,
            "improvement": improvement,
            "n_early": len(early),
            "n_late": len(late),
            "n_total": len(values),
        }

    # ------------------------------------------------------------------
    # Multi-episode aggregation
    # ------------------------------------------------------------------

    def aggregate_reward_trajectories(
        self, trajectories: list[list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Average reward trajectories across multiple episodes.

        Aligns by window index; returns mean + std per window.
        """
        if not trajectories:
            return []

        max_windows = max(len(t) for t in trajectories)
        aggregated: list[dict[str, Any]] = []

        for w in range(max_windows):
            rewards = []
            for traj in trajectories:
                if w < len(traj):
                    rewards.append(traj[w].get("mean_reward", 0.0))
            if not rewards:
                continue
            aggregated.append({
                "window": w,
                "mean_reward": _mean(rewards),
                "std_reward": _std(rewards),
                "n_episodes": len(rewards),
            })
        return aggregated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _linreg(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Simple least-squares linear regression.  Returns (slope, intercept)."""
    n = len(xs)
    if n < 2:
        return (0.0, ys[0] if ys else 0.0)
    mx = _mean(xs)
    my = _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if abs(den) < 1e-12:
        return (0.0, my)
    slope = num / den
    intercept = my - slope * mx
    return (slope, intercept)
