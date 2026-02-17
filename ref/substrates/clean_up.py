"""
Clean Up -- DigiSoup substrate replicating Melting Pot dynamics.

Public goods / free-rider dilemma:  the grid is split into a RIVER zone
(left third) and an ORCHARD zone (right two-thirds).

  * Pollution spawns in the river at a steady rate.
  * Apples grow in the orchard, but only when pollution is low.
  * Agents can CLEAN pollution (altruistic: benefits everyone) or
    HARVEST apples (selfish: benefits only the harvester).

The tension: everyone wants apples, but if nobody cleans the river,
apple growth collapses for all.

PettingZoo AEC API.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import GridWorldBaseEnv, AGENT_PALETTE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Cell types
CELL_EMPTY     = 0
CELL_POLLUTION = 1
CELL_APPLE     = 2

# Zone markers (not stored in grid, derived from column index)
# River: columns [0, river_boundary)
# Orchard: columns [river_boundary, grid_cols)

POLLUTION_SPAWN_PROB = 0.02   # per empty river cell per tick
APPLE_BASE_REGROW   = 0.01   # max apple regrowth probability (when no pollution)
CLEAN_RADIUS         = 1     # 3x3 cleaning area (radius 1 around agent)

INITIAL_APPLE_DENSITY    = 0.20
INITIAL_POLLUTION_DENSITY = 0.05

# Colours (RGB)
COLOR_RIVER_BG    = (30, 30, 80)
COLOR_POLLUTION   = (128, 128, 128)
COLOR_ORCHARD_BG  = (60, 40, 20)
COLOR_APPLE       = (0, 200, 0)


class CleanUpEnv(GridWorldBaseEnv):
    """Clean Up environment (PettingZoo AEC)."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "clean_up_v0",
    }

    def __init__(
        self,
        n_agents: int = 7,
        grid_rows: int = 18,
        grid_cols: int = 25,
        max_cycles: int = 1000,
        render_mode: Optional[str] = None,
    ):
        super().__init__(
            n_agents=n_agents,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            max_cycles=max_cycles,
            render_mode=render_mode,
        )
        self.river_boundary = grid_cols // 3  # left third is river
        self.grid: np.ndarray = np.zeros((grid_rows, grid_cols), dtype=np.int8)

    # ------------------------------------------------------------------
    # Zone queries
    # ------------------------------------------------------------------
    def _is_river(self, c: int) -> bool:
        return c < self.river_boundary

    def _is_orchard(self, c: int) -> bool:
        return c >= self.river_boundary

    # ------------------------------------------------------------------
    # World setup
    # ------------------------------------------------------------------
    def _reset_world(self) -> None:
        self.grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int8)

        # Seed some initial pollution in the river
        for r in range(self.grid_rows):
            for c in range(self.river_boundary):
                if self._rng.random() < INITIAL_POLLUTION_DENSITY:
                    self.grid[r, c] = CELL_POLLUTION

        # Seed apples in the orchard
        for r in range(self.grid_rows):
            for c in range(self.river_boundary, self.grid_cols):
                if self._rng.random() < INITIAL_APPLE_DENSITY:
                    self.grid[r, c] = CELL_APPLE

        self._scatter_agents()

    # ------------------------------------------------------------------
    # Movement hook -- harvest apples by walking onto them
    # ------------------------------------------------------------------
    def _on_agent_move(self, agent: str, r: int, c: int) -> None:
        if self._is_orchard(c) and self.grid[r, c] == CELL_APPLE:
            self.grid[r, c] = CELL_EMPTY
            self.rewards[agent] += 1.0

    # ------------------------------------------------------------------
    # Interaction -- clean pollution if in river, harvest if on apple
    # ------------------------------------------------------------------
    def _process_interaction(self, agent: str) -> None:
        r, c = self.agent_positions[agent]

        if self._is_river(c):
            # Clean: remove pollution in 3x3 area
            for dr in range(-CLEAN_RADIUS, CLEAN_RADIUS + 1):
                for dc in range(-CLEAN_RADIUS, CLEAN_RADIUS + 1):
                    nr, nc = r + dr, c + dc
                    if (
                        self._in_bounds(nr, nc)
                        and self._is_river(nc)
                        and self.grid[nr, nc] == CELL_POLLUTION
                    ):
                        self.grid[nr, nc] = CELL_EMPTY
        else:
            # In orchard -- harvest apple at current cell
            if self.grid[r, c] == CELL_APPLE:
                self.grid[r, c] = CELL_EMPTY
                self.rewards[agent] += 1.0

    # ------------------------------------------------------------------
    # World tick
    # ------------------------------------------------------------------
    def _tick_world(self) -> None:
        rows, cols = self.grid_rows, self.grid_cols
        rb = self.river_boundary

        # --- Pollution spawns in river (vectorised) ---
        river_slice = self.grid[:, :rb]
        empty_river = river_slice == CELL_EMPTY
        spawn_roll = self._rng.random((rows, rb))
        new_pollution = empty_river & (spawn_roll < POLLUTION_SPAWN_PROB)
        river_slice[new_pollution] = CELL_POLLUTION

        # --- Apple regrowth in orchard (vectorised) ---
        total_river_cells = rows * rb
        if total_river_cells == 0:
            pollution_ratio = 0.0
        else:
            pollution_ratio = float(np.count_nonzero(
                self.grid[:, :rb] == CELL_POLLUTION
            )) / total_river_cells

        apple_prob = max(0.0, 1.0 - pollution_ratio) * APPLE_BASE_REGROW

        if apple_prob > 0:
            orchard_slice = self.grid[:, rb:]
            empty_orchard = orchard_slice == CELL_EMPTY
            apple_roll = self._rng.random((rows, cols - rb))
            new_apples = empty_orchard & (apple_roll < apple_prob)
            orchard_slice[new_apples] = CELL_APPLE

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_cell(self, r: int, c: int) -> tuple[int, int, int]:
        val = self.grid[r, c]
        if val == CELL_POLLUTION:
            return COLOR_POLLUTION
        if val == CELL_APPLE:
            return COLOR_APPLE

        if self._is_river(c):
            return COLOR_RIVER_BG
        return COLOR_ORCHARD_BG

    def _get_agent_color(self, agent: str) -> tuple[int, int, int]:
        idx = self.possible_agents.index(agent)
        return AGENT_PALETTE[idx % len(AGENT_PALETTE)]


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = CleanUpEnv(n_agents=4, grid_rows=12, grid_cols=18, max_cycles=50,
                     render_mode="rgb_array")
    env.reset(seed=42)

    total_steps = 0
    while env.agents:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
        else:
            action = env.action_space(agent).sample()
            env.step(action)
        total_steps += 1

    print(f"Episode finished after {total_steps} steps, {env._cycle_count} cycles")
    print("Rewards:", {a: round(r, 1) for a, r in env.episode_rewards.items()})
    img = env.render()
    print(f"Render shape: {img.shape}")
    env.close()
    print("clean_up_v0: smoke test PASSED")
