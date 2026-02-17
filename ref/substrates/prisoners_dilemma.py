"""
Prisoner's Dilemma in the Matrix -- DigiSoup substrate replicating Melting Pot.

Agents roam a grid freely.  When two agents are adjacent and one uses the
INTERACT action, a one-shot Prisoner's Dilemma is played between them.

  * INTERACT  = cooperate
  * any other action while adjacent = defect  (evaluated at resolution time)

Each agent's most recent choice is encoded as a visible colour:
  blue  = cooperated last
  red   = defected last
  gray  = no interaction yet

This lets agents develop conditional strategies (e.g., tit-for-tat by
observing the other's colour before deciding to interact).

Payoff matrix (row = self, col = other):
          Cooperate   Defect
  Coop      3, 3       0, 5
  Defect    5, 0       1, 1

Scattered yellow resource tokens give +0.5 on collection, providing a
baseline movement incentive even when no PD encounters occur.

PettingZoo AEC API.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import (
    GridWorldBaseEnv,
    AGENT_PALETTE,
    DIR_DELTAS,
)


# ---------------------------------------------------------------------------
# PD payoff matrix  [my_action][their_action]  -> my_reward
# ---------------------------------------------------------------------------
#               other cooperates   other defects
PD_PAYOFFS = {
    ("C", "C"): (3.0, 3.0),
    ("C", "D"): (0.0, 5.0),
    ("D", "C"): (5.0, 0.0),
    ("D", "D"): (1.0, 1.0),
}

# Interaction state colours
COLOR_COOPERATOR = (80, 120, 255)   # blue
COLOR_DEFECTOR   = (255, 60, 60)    # red
COLOR_NEUTRAL    = (150, 150, 150)  # gray

# Resource
COLOR_RESOURCE   = (200, 200, 0)    # yellow
COLOR_BG         = (30, 30, 30)

RESOURCE_DENSITY = 0.04       # fraction of cells with resources at start
RESOURCE_RESPAWN = 0.005      # probability per empty cell per tick
RESOURCE_REWARD  = 0.5


class PrisonersDilemmaEnv(GridWorldBaseEnv):
    """Prisoner's Dilemma in the Matrix (PettingZoo AEC)."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "prisoners_dilemma_v0",
    }

    def __init__(
        self,
        n_agents: int = 8,
        grid_size: int = 20,
        max_cycles: int = 1000,
        render_mode: Optional[str] = None,
    ):
        super().__init__(
            n_agents=n_agents,
            grid_rows=grid_size,
            grid_cols=grid_size,
            max_cycles=max_cycles,
            render_mode=render_mode,
        )
        # Resource grid: 0 = empty, 1 = resource present
        self.resource_grid: np.ndarray = np.zeros(
            (grid_size, grid_size), dtype=np.int8
        )

        # Per-agent PD state
        # "C" = cooperated last, "D" = defected last, None = no interaction
        self.pd_state: dict[str, str | None] = {}

        # Pending interactions for the current cycle
        # agent -> True if agent used INTERACT this step
        self._interact_flag: dict[str, bool] = {}

        # Track PD stats
        self.pd_stats: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------
    # World setup
    # ------------------------------------------------------------------
    def _reset_world(self) -> None:
        self.resource_grid = np.zeros(
            (self.grid_rows, self.grid_cols), dtype=np.int8
        )
        # Scatter resources
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if self._rng.random() < RESOURCE_DENSITY:
                    self.resource_grid[r, c] = 1

        self.pd_state = {a: None for a in self.agents}
        self._interact_flag = {a: False for a in self.agents}
        self.pd_stats = {
            a: {"cooperations": 0, "defections": 0, "encounters": 0}
            for a in self.agents
        }

        self._scatter_agents()

    # ------------------------------------------------------------------
    # Movement -- collect resources
    # ------------------------------------------------------------------
    def _on_agent_move(self, agent: str, r: int, c: int) -> None:
        if self.resource_grid[r, c] == 1:
            self.resource_grid[r, c] = 0
            self.rewards[agent] += RESOURCE_REWARD

    # ------------------------------------------------------------------
    # Interaction -- flag intent to cooperate; resolution at end of cycle
    # ------------------------------------------------------------------
    def _process_interaction(self, agent: str) -> None:
        # Mark this agent as having used INTERACT.
        # Actual PD resolution happens in _resolve_pd_encounters at tick time,
        # but we also do an immediate check: if an adjacent agent has already
        # flagged interact this cycle, resolve now.
        self._interact_flag[agent] = True

        # Check for adjacent agents who also have their flag set
        ar, ac = self.agent_positions[agent]
        for other in self.agents:
            if other == agent:
                continue
            if self.terminations.get(other, False):
                continue
            orow, ocol = self.agent_positions[other]
            if abs(ar - orow) + abs(ac - ocol) == 1:
                # Adjacent -- resolve PD immediately for this pair
                self._resolve_pd_pair(agent, other)

    def _resolve_pd_pair(self, a1: str, a2: str) -> None:
        """Resolve a single PD encounter between two adjacent agents."""
        choice_a1 = "C" if self._interact_flag.get(a1, False) else "D"
        choice_a2 = "C" if self._interact_flag.get(a2, False) else "D"

        r1, r2 = PD_PAYOFFS[(choice_a1, choice_a2)]
        self.rewards[a1] += r1
        self.rewards[a2] += r2

        # Update state colours
        self.pd_state[a1] = choice_a1
        self.pd_state[a2] = choice_a2

        # Stats
        self.pd_stats[a1]["encounters"] += 1
        self.pd_stats[a2]["encounters"] += 1
        if choice_a1 == "C":
            self.pd_stats[a1]["cooperations"] += 1
        else:
            self.pd_stats[a1]["defections"] += 1
        if choice_a2 == "C":
            self.pd_stats[a2]["cooperations"] += 1
        else:
            self.pd_stats[a2]["defections"] += 1

        # Clear flags so pair isn't resolved again this cycle
        self._interact_flag[a1] = False
        self._interact_flag[a2] = False

    # ------------------------------------------------------------------
    # World tick
    # ------------------------------------------------------------------
    def _tick_world(self) -> None:
        # Resolve any remaining PD encounters for agents whose partner
        # didn't explicitly interact (those partners default to defect).
        for agent in self.agents:
            if not self._interact_flag.get(agent, False):
                continue
            # Agent interacted but wasn't paired yet -- find adjacent agents
            ar, ac = self.agent_positions[agent]
            for other in self.agents:
                if other == agent:
                    continue
                if self.terminations.get(other, False):
                    continue
                orow, ocol = self.agent_positions[other]
                if abs(ar - orow) + abs(ac - ocol) == 1:
                    self._resolve_pd_pair(agent, other)
                    break  # one encounter per agent per cycle

        # Clear all interact flags for next cycle
        for a in self.agents:
            self._interact_flag[a] = False

        # Respawn resources
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if self.resource_grid[r, c] == 0:
                    if self._rng.random() < RESOURCE_RESPAWN:
                        self.resource_grid[r, c] = 1

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_cell(self, r: int, c: int) -> tuple[int, int, int]:
        if self.resource_grid[r, c] == 1:
            return COLOR_RESOURCE
        return COLOR_BG

    def _get_agent_color(self, agent: str) -> tuple[int, int, int]:
        state = self.pd_state.get(agent)
        if state == "C":
            return COLOR_COOPERATOR
        elif state == "D":
            return COLOR_DEFECTOR
        return COLOR_NEUTRAL


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = PrisonersDilemmaEnv(
        n_agents=4, grid_size=12, max_cycles=50, render_mode="rgb_array"
    )
    env.reset(seed=42)

    total_steps = 0
    while env.agents:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
        else:
            # Bias toward INTERACT to trigger some PD encounters
            if env._rng.random() < 0.3:
                action = 7  # interact
            else:
                action = env.action_space(agent).sample()
            env.step(action)
        total_steps += 1

    print(f"Episode finished after {total_steps} steps, {env._cycle_count} cycles")
    print("Rewards:", {a: round(r, 1) for a, r in env.episode_rewards.items()})
    print("PD stats:", env.pd_stats)
    img = env.render()
    print(f"Render shape: {img.shape}")
    env.close()
    print("prisoners_dilemma_v0: smoke test PASSED")
