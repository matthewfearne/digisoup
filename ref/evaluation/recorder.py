# recorder.py -- Record per-step observations, actions, rewards, state.
"""
Detailed per-step data recording for post-hoc analysis.

Saves step-level data (observation entropy, reward, action, agent internal
state snapshot) to JSON for cooperation timeline, role timeline, and
temporal analysis.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any


class EpisodeRecorder:
    """Records detailed per-step data for a single episode."""

    def __init__(
        self,
        substrate_name: str,
        agent_type: str,
        episode_num: int,
    ):
        self.substrate_name = substrate_name
        self.agent_type = agent_type
        self.episode_num = episode_num
        self.steps: list[dict[str, Any]] = []
        self.start_time = time.time()
        self.metadata: dict[str, Any] = {
            "substrate": substrate_name,
            "agent_type": agent_type,
            "episode": episode_num,
            "start_time": self.start_time,
        }

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        step: int,
        agent_id: str,
        observation_entropy: float,
        reward: float,
        action: int,
        agent_state: dict[str, Any] | None = None,
    ) -> None:
        """Record one agent-step."""
        entry: dict[str, Any] = {
            "step": step,
            "agent_id": agent_id,
            "observation_entropy": observation_entropy,
            "reward": reward,
            "action": action,
        }
        if agent_state:
            # Capture a snapshot of the agent's internal state.  Only
            # include serialisable scalar/string fields.
            safe_state: dict[str, Any] = {}
            for k, v in agent_state.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    safe_state[k] = v
                elif isinstance(v, list) and all(
                    isinstance(x, (int, float, str, bool, type(None)))
                    for x in v
                ):
                    safe_state[k] = v
            entry["agent_state"] = safe_state
        self.steps.append(entry)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, results_dir: str) -> str:
        """Save recorded data to a JSON file.

        Returns the path to the saved file.
        """
        os.makedirs(results_dir, exist_ok=True)
        filename = (
            f"{self.substrate_name}_{self.agent_type}"
            f"_ep{self.episode_num:03d}.json"
        )
        path = os.path.join(results_dir, filename)
        payload = {
            "metadata": self.metadata,
            "n_steps": len(self.steps),
            "steps": self.steps,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh)
        return path

    # ------------------------------------------------------------------
    # Derived timelines
    # ------------------------------------------------------------------

    def get_cooperation_timeline(
        self, window_size: int = 50
    ) -> list[dict[str, Any]]:
        """Return cooperation ratio over time, binned into windows.

        Each entry: {"window": i, "step_start": ..., "step_end": ...,
                      "cooperation_ratio": float, "n_interactions": int}.
        """
        from src.evaluation.metrics import INTERACT_ACTION

        interact_steps = [
            s for s in self.steps if s["action"] == INTERACT_ACTION
        ]
        if not interact_steps:
            return []

        timeline: list[dict[str, Any]] = []
        for i in range(0, len(interact_steps), window_size):
            window = interact_steps[i : i + window_size]
            coop = sum(1 for s in window if s["reward"] >= 0)
            total = len(window)
            timeline.append({
                "window": i // window_size,
                "step_start": window[0]["step"],
                "step_end": window[-1]["step"],
                "cooperation_ratio": coop / total if total else 0.0,
                "n_interactions": total,
            })
        return timeline

    def get_role_timeline(
        self, window_size: int = 100
    ) -> list[dict[str, Any]]:
        """Return role distribution over time from agent_state snapshots.

        Requires that agent_state["role"] was recorded.
        """
        steps_with_role = [
            s for s in self.steps
            if s.get("agent_state", {}).get("role") is not None
        ]
        if not steps_with_role:
            return []

        timeline: list[dict[str, Any]] = []
        for i in range(0, len(steps_with_role), window_size):
            window = steps_with_role[i : i + window_size]
            role_counts: dict[str, int] = {}
            for s in window:
                role = s["agent_state"]["role"]
                role_counts[role] = role_counts.get(role, 0) + 1
            total = len(window)
            role_fracs = {r: c / total for r, c in role_counts.items()}
            timeline.append({
                "window": i // window_size,
                "step_start": window[0]["step"],
                "step_end": window[-1]["step"],
                "role_distribution": role_fracs,
                "n_samples": total,
            })
        return timeline

    def get_reward_timeline(
        self, window_size: int = 50
    ) -> list[dict[str, Any]]:
        """Return mean reward per window over time."""
        if not self.steps:
            return []

        timeline: list[dict[str, Any]] = []
        for i in range(0, len(self.steps), window_size):
            window = self.steps[i : i + window_size]
            rewards = [s["reward"] for s in window]
            mean_r = sum(rewards) / len(rewards) if rewards else 0.0
            timeline.append({
                "window": i // window_size,
                "step_start": window[0]["step"],
                "step_end": window[-1]["step"],
                "mean_reward": mean_r,
                "total_reward": sum(rewards),
                "n_steps": len(window),
            })
        return timeline


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_recording(path: str) -> dict[str, Any]:
    """Load a saved recording JSON file."""
    with open(path) as fh:
        return json.load(fh)
