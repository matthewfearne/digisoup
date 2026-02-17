# emergence.py -- Detect role specialization, spatial patterns, 98/2 ratio.
"""
Behavioral emergence analysis.

Analyses the emergent behavioural types from DigiSoup agents:
    - Role distribution at episode end
    - Role shifts over episode time
    - Check for DigiSoup's signature 98/2 cooperation ratio
    - Comparison of type diversity (Shannon entropy of role distribution)
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any


# ---------------------------------------------------------------------------
# DigiSoup's 12 emergent behavioural types (from the original paper).
# Not all may appear in Melting Pot -- we track whatever roles the agent
# reports and classify them.
# ---------------------------------------------------------------------------

DIGISOUP_TYPES = [
    "cooperator",
    "defector",
    "explorer",
    "scavenger",
    "defender",
    "hoarder",
    "altruist",
    "reciprocator",
    "opportunist",
    "neutral",
    "predator",
    "parasite",
]


class EmergenceAnalyzer:
    """Analyzes emergent behavioral types (the 12 DigiSoup types)."""

    # ------------------------------------------------------------------
    # Role distribution
    # ------------------------------------------------------------------

    def role_distribution(
        self, data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Distribution of roles at the end of each episode.

        *data* is a list of per-episode result dicts.  Each must contain
        a ``final_roles`` key: dict[agent_id, role_name].

        If *data* is a list of step-level recordings, we take the last
        recorded role per agent.

        Returns:
            dict with keys: counts (Counter), fractions (dict[str, float]),
            diversity (Shannon entropy of distribution), n_types (int),
            dominant_type (str).
        """
        role_counts: Counter = Counter()

        for entry in data:
            # Accept either aggregated or step-level data
            if "final_roles" in entry:
                for role in entry["final_roles"].values():
                    role_counts[role] += 1
            elif "agent_state" in entry:
                role = entry.get("agent_state", {}).get("role")
                if role is not None:
                    role_counts[role] += 1
            elif "steps" in entry:
                # Full recording: extract last role per agent
                last_roles = _extract_final_roles(entry["steps"])
                role_counts.update(last_roles.values())

        total = sum(role_counts.values())
        fractions = {
            role: count / total
            for role, count in role_counts.items()
        } if total > 0 else {}

        diversity = _shannon_entropy(list(fractions.values()))
        n_types = len(role_counts)
        dominant = role_counts.most_common(1)[0][0] if role_counts else "none"

        return {
            "counts": dict(role_counts),
            "fractions": fractions,
            "diversity": diversity,
            "n_types": n_types,
            "dominant_type": dominant,
            "total_samples": total,
        }

    # ------------------------------------------------------------------
    # Role timeline
    # ------------------------------------------------------------------

    def role_timeline(
        self,
        data: list[dict[str, Any]],
        window_size: int = 100,
    ) -> list[dict[str, Any]]:
        """How roles shift over episode time.

        *data* is a list of step-level dicts with agent_state.role.

        Returns a list of window dicts, each containing the role
        distribution for that window.
        """
        steps_with_role = [
            s for s in data
            if s.get("agent_state", {}).get("role") is not None
        ]
        if not steps_with_role:
            return []

        timeline: list[dict[str, Any]] = []
        for i in range(0, len(steps_with_role), window_size):
            window = steps_with_role[i : i + window_size]
            counts: Counter = Counter()
            for s in window:
                counts[s["agent_state"]["role"]] += 1
            total = sum(counts.values())
            fracs = {r: c / total for r, c in counts.items()} if total else {}
            timeline.append({
                "window": i // window_size,
                "step_start": window[0].get("step", i),
                "step_end": window[-1].get("step", i + len(window)),
                "role_distribution": fracs,
                "diversity": _shannon_entropy(list(fracs.values())),
                "n_types": len(counts),
                "n_samples": total,
            })
        return timeline

    # ------------------------------------------------------------------
    # 98/2 ratio check
    # ------------------------------------------------------------------

    def check_98_2_ratio(
        self, data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Check for DigiSoup's signature 98/2 cooperation ratio.

        In the original DigiSoup experiment, approximately 98 % of agents
        evolved into cooperative types and 2 % into defectors.

        *data* is a list of per-episode metric dicts with
        ``cooperation_ratio`` keys.

        Returns:
            dict with mean_cooperation, matches_98_2 (bool),
            deviation_from_98 (float), n_episodes.
        """
        ratios = [
            ep.get("cooperation_ratio", 0.0) for ep in data
            if "cooperation_ratio" in ep
        ]
        if not ratios:
            return {
                "mean_cooperation": 0.0,
                "matches_98_2": False,
                "deviation_from_98": 1.0,
                "n_episodes": 0,
            }

        mean_coop = sum(ratios) / len(ratios)
        target = 0.98
        deviation = abs(mean_coop - target)
        # Consider it a match if within 5 percentage points
        matches = deviation <= 0.05

        return {
            "mean_cooperation": mean_coop,
            "matches_98_2": matches,
            "deviation_from_98": deviation,
            "target": target,
            "n_episodes": len(ratios),
        }

    # ------------------------------------------------------------------
    # Type transition matrix
    # ------------------------------------------------------------------

    def type_transitions(
        self, data: list[dict[str, Any]]
    ) -> dict[str, dict[str, int]]:
        """Compute a role transition matrix from step-level data.

        Returns a nested dict: transitions[from_role][to_role] = count.
        """
        transitions: dict[str, dict[str, int]] = {}

        # Group by agent_id, sort by step
        by_agent: dict[str, list[dict[str, Any]]] = {}
        for s in data:
            aid = s.get("agent_id", "unknown")
            if aid not in by_agent:
                by_agent[aid] = []
            by_agent[aid].append(s)

        for aid, steps in by_agent.items():
            steps.sort(key=lambda s: s.get("step", 0))
            prev_role = None
            for s in steps:
                role = s.get("agent_state", {}).get("role")
                if role is None:
                    continue
                if prev_role is not None and role != prev_role:
                    if prev_role not in transitions:
                        transitions[prev_role] = {}
                    transitions[prev_role][role] = (
                        transitions[prev_role].get(role, 0) + 1
                    )
                prev_role = role

        return transitions

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(
        self, episode_metrics: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate a full emergence summary from episode-level metrics."""
        ratio_check = self.check_98_2_ratio(episode_metrics)

        # Collect final roles if available
        dist = self.role_distribution(episode_metrics)

        return {
            "role_distribution": dist,
            "ratio_check_98_2": ratio_check,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(probs: list[float]) -> float:
    """Shannon entropy in bits from a list of probabilities."""
    if not probs:
        return 0.0
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log2(p)
    return h


def _extract_final_roles(
    steps: list[dict[str, Any]],
) -> dict[str, str]:
    """From a list of step dicts, extract the last role per agent."""
    last: dict[str, str] = {}
    for s in steps:
        aid = s.get("agent_id", "unknown")
        role = s.get("agent_state", {}).get("role")
        if role is not None:
            last[aid] = role
    return last
