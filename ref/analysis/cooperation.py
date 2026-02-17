# cooperation.py -- Cooperation detection and measurement from episode data.
"""
Cooperation metric analysis.

Analyses cooperation patterns across episodes:
    - Growth rate of cooperation over episode time
    - Comparison between DigiSoup and random baselines
    - Correlation between behavioural types and cooperation
"""
from __future__ import annotations

import math
from typing import Any


class CooperationAnalyzer:
    """Analyzes cooperation patterns across episodes."""

    # ------------------------------------------------------------------
    # Growth rate
    # ------------------------------------------------------------------

    def cooperation_growth_rate(
        self, timeline: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate how cooperation changes over episode time.

        *timeline* is a list of dicts as returned by
        ``EpisodeRecorder.get_cooperation_timeline()``.  Each entry must
        have ``cooperation_ratio`` and ``window`` fields.

        Returns:
            dict with keys: slope (float), intercept (float),
            initial_ratio (float), final_ratio (float), delta (float),
            n_windows (int).  *slope* is the least-squares fit.
        """
        if not timeline:
            return {
                "slope": 0.0,
                "intercept": 0.0,
                "initial_ratio": 0.0,
                "final_ratio": 0.0,
                "delta": 0.0,
                "n_windows": 0,
            }

        xs = [float(t["window"]) for t in timeline]
        ys = [float(t["cooperation_ratio"]) for t in timeline]
        n = len(xs)

        initial = ys[0]
        final = ys[-1]
        delta = final - initial

        # Least-squares linear regression
        slope, intercept = _linreg(xs, ys)

        return {
            "slope": slope,
            "intercept": intercept,
            "initial_ratio": initial,
            "final_ratio": final,
            "delta": delta,
            "n_windows": n,
        }

    # ------------------------------------------------------------------
    # DigiSoup vs Random comparison
    # ------------------------------------------------------------------

    def cooperation_vs_random(
        self,
        digisoup_data: list[dict[str, Any]],
        random_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare cooperation levels: DigiSoup vs random baseline.

        Each input is a list of per-episode metric dicts (output of
        ``MetricsCollector.compute()``).

        Returns:
            dict with mean_digisoup, mean_random, absolute_diff,
            relative_improvement (%), p_value_approx.
        """
        ds_ratios = [
            ep.get("cooperation_ratio", 0.0) for ep in digisoup_data
        ]
        rn_ratios = [
            ep.get("cooperation_ratio", 0.0) for ep in random_data
        ]

        mean_ds = _mean(ds_ratios)
        mean_rn = _mean(rn_ratios)
        abs_diff = mean_ds - mean_rn
        rel_imp = (
            (abs_diff / mean_rn * 100.0) if abs_diff != 0 and mean_rn > 1e-8
            else 0.0
        )

        # Approximate p-value via Welch's t-test
        p_val = _welch_t_pvalue(ds_ratios, rn_ratios)

        return {
            "mean_digisoup": mean_ds,
            "mean_random": mean_rn,
            "absolute_diff": abs_diff,
            "relative_improvement_pct": rel_imp,
            "p_value_approx": p_val,
            "n_digisoup": len(ds_ratios),
            "n_random": len(rn_ratios),
        }

    # ------------------------------------------------------------------
    # Type-cooperation correlation
    # ------------------------------------------------------------------

    def type_cooperation_correlation(
        self, data: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """Determine which behavioural types cooperate most.

        *data* is a list of step-level dicts (from a recording) containing
        ``action`` and ``agent_state.role``.

        Returns a dict mapping role name to {cooperation_ratio, n_actions,
        n_interactions}.
        """
        from src.evaluation.metrics import INTERACT_ACTION

        role_stats: dict[str, dict[str, int]] = {}

        for step in data:
            state = step.get("agent_state", {})
            role = state.get("role", "unknown") if state else "unknown"
            if role not in role_stats:
                role_stats[role] = {"coop": 0, "defect": 0, "total": 0}
            role_stats[role]["total"] += 1

            if step.get("action") == INTERACT_ACTION:
                if step.get("reward", 0.0) >= 0:
                    role_stats[role]["coop"] += 1
                else:
                    role_stats[role]["defect"] += 1

        result: dict[str, dict[str, float]] = {}
        for role, counts in role_stats.items():
            interactions = counts["coop"] + counts["defect"]
            ratio = (
                counts["coop"] / interactions if interactions > 0 else 0.0
            )
            result[role] = {
                "cooperation_ratio": ratio,
                "n_actions": counts["total"],
                "n_interactions": interactions,
            }
        return result

    # ------------------------------------------------------------------
    # Multi-episode aggregate
    # ------------------------------------------------------------------

    def aggregate_timelines(
        self, timelines: list[list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Average cooperation timelines across multiple episodes.

        Aligns by window index and computes mean + std of cooperation_ratio.
        """
        if not timelines:
            return []

        max_windows = max(len(t) for t in timelines)
        aggregated: list[dict[str, Any]] = []

        for w in range(max_windows):
            ratios = []
            for tl in timelines:
                if w < len(tl):
                    ratios.append(tl[w]["cooperation_ratio"])
            if not ratios:
                continue
            aggregated.append({
                "window": w,
                "cooperation_ratio_mean": _mean(ratios),
                "cooperation_ratio_std": _std(ratios),
                "n_episodes": len(ratios),
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


def _welch_t_pvalue(
    a: list[float], b: list[float]
) -> float:
    """Approximate two-sided p-value from Welch's t-test.

    Uses the normal approximation for simplicity (valid for n >= 20).
    Returns 1.0 if the test cannot be computed.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 1.0
    ma, mb = _mean(a), _mean(b)
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se < 1e-12:
        return 0.0 if abs(ma - mb) > 1e-12 else 1.0
    t_stat = (ma - mb) / se

    # Two-sided p-value approximation using the error function
    # P(|T| > |t|) ~ 2 * (1 - Phi(|t|)) for large df
    try:
        p = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    except Exception:
        p = 1.0
    return max(0.0, min(1.0, p))


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))
