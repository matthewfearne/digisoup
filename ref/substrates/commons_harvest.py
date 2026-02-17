"""
Commons Harvest Open -- DigiSoup substrate replicating Melting Pot dynamics.

Tragedy of the commons:  renewable green apples grow back only if local
density stays above a critical threshold.  Overharvesting a region kills
regrowth permanently, creating a collective-action dilemma.

PettingZoo AEC API.

Grid cells
    0  EMPTY
    1  GROWING  (will become RIPE after GROW_TICKS steps)
    2  RIPE     (harvestable, yields +1 reward)

Regrowth rule
    An EMPTY cell turns GROWING with probability REGROW_PROB each tick
    *only if* at least 1 of its 8 neighbours is GROWING or RIPE.
    However, if every cell in the surrounding 5x5 patch is EMPTY, regrowth
    is permanently blocked (tragedy of the commons).

    A GROWING cell becomes RIPE after GROW_TICKS ticks.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import GridWorldBaseEnv, AGENT_PALETTE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CELL_EMPTY   = 0
CELL_GROWING = 1
CELL_RIPE    = 2

REGROW_PROB  = 0.01   # per-tick probability an EMPTY cell starts growing
GROW_TICKS   = 20     # ticks for GROWING -> RIPE
DEPLETION_RADIUS = 2  # half-side of the 5x5 depletion check window

INITIAL_RESOURCE_DENSITY = 0.15  # fraction of cells seeded as RIPE at start

# Colours (RGB)
COLOR_BG        = (40, 40, 40)
COLOR_GROWING   = (100, 200, 100)
COLOR_RIPE      = (0, 255, 0)


class CommonsHarvestEnv(GridWorldBaseEnv):
    """Commons Harvest Open environment (PettingZoo AEC)."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "commons_harvest_v0",
    }

    def __init__(
        self,
        n_agents: int = 7,
        grid_size: int = 25,
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
        # Grid state
        self.resource_grid: np.ndarray = np.zeros(
            (grid_size, grid_size), dtype=np.int8
        )
        # Tracks how many ticks a GROWING cell has been growing
        self.grow_timer: np.ndarray = np.zeros(
            (grid_size, grid_size), dtype=np.int32
        )

    # ------------------------------------------------------------------
    # World setup
    # ------------------------------------------------------------------
    def _reset_world(self) -> None:
        self.resource_grid = np.zeros(
            (self.grid_rows, self.grid_cols), dtype=np.int8
        )
        self.grow_timer = np.zeros(
            (self.grid_rows, self.grid_cols), dtype=np.int32
        )

        # Seed initial resources in clustered patches
        n_seeds = max(
            1, int(self.grid_rows * self.grid_cols * INITIAL_RESOURCE_DENSITY)
        )
        # Place several cluster centres and fill nearby cells
        n_clusters = max(3, n_seeds // 10)
        centres = [
            (
                int(self._rng.integers(2, self.grid_rows - 2)),
                int(self._rng.integers(2, self.grid_cols - 2)),
            )
            for _ in range(n_clusters)
        ]
        placed = 0
        for cr, cc in centres:
            radius = self._rng.integers(2, 5)
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    r, c = cr + dr, cc + dc
                    if (
                        0 <= r < self.grid_rows
                        and 0 <= c < self.grid_cols
                        and self.resource_grid[r, c] == CELL_EMPTY
                        and placed < n_seeds
                    ):
                        if self._rng.random() < 0.6:
                            self.resource_grid[r, c] = CELL_RIPE
                            placed += 1

        # Scatter agents
        self._scatter_agents()

    # ------------------------------------------------------------------
    # Movement hook -- harvest on step
    # ------------------------------------------------------------------
    def _on_agent_move(self, agent: str, r: int, c: int) -> None:
        if self.resource_grid[r, c] == CELL_RIPE:
            self.resource_grid[r, c] = CELL_EMPTY
            self.grow_timer[r, c] = 0
            self.rewards[agent] += 1.0

    # ------------------------------------------------------------------
    # Interaction -- also harvest the cell the agent stands on
    # ------------------------------------------------------------------
    def _process_interaction(self, agent: str) -> None:
        r, c = self.agent_positions[agent]
        if self.resource_grid[r, c] == CELL_RIPE:
            self.resource_grid[r, c] = CELL_EMPTY
            self.grow_timer[r, c] = 0
            self.rewards[agent] += 1.0

    # ------------------------------------------------------------------
    # World tick (once per cycle -- resource dynamics)
    # ------------------------------------------------------------------
    def _tick_world(self) -> None:
        rows, cols = self.grid_rows, self.grid_cols

        # --- Advance GROWING -> RIPE ---
        growing_mask = self.resource_grid == CELL_GROWING
        self.grow_timer[growing_mask] += 1
        ripe_mask = growing_mask & (self.grow_timer >= GROW_TICKS)
        self.resource_grid[ripe_mask] = CELL_RIPE
        self.grow_timer[ripe_mask] = 0

        # --- Regrowth: EMPTY -> GROWING ---
        empty_mask = self.resource_grid == CELL_EMPTY

        # Count neighbours that are GROWING or RIPE (8-connected)
        # Using padded array to avoid roll artefacts
        alive = (self.resource_grid >= CELL_GROWING).astype(np.float32)
        padded = np.pad(alive, 1, mode="constant", constant_values=0)
        neighbour_count = (
            padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:]
            + padded[1:-1, 0:-2]                       + padded[1:-1, 2:]
            + padded[2:,   0:-2] + padded[2:,   1:-1] + padded[2:,   2:]
        )

        # Depletion check: if every cell in the 5x5 window is EMPTY,
        # regrowth is blocked.  Vectorised via padded sliding-window sum.
        non_empty = (self.resource_grid > CELL_EMPTY).astype(np.float32)
        k = DEPLETION_RADIUS  # 2 -> 5x5 window
        padded_ne = np.pad(non_empty, k, mode="constant", constant_values=0)
        # Build a 5x5 box sum by shifting and accumulating
        window_sum = np.zeros((rows, cols), dtype=np.float32)
        for dr in range(2 * k + 1):
            for dc in range(2 * k + 1):
                window_sum += padded_ne[dr:dr + rows, dc:dc + cols]
        can_regrow = window_sum > 0

        # Apply regrowth probability
        regrow_candidates = empty_mask & (neighbour_count >= 1) & can_regrow
        rand_vals = self._rng.random((rows, cols))
        new_growing = regrow_candidates & (rand_vals < REGROW_PROB)
        self.resource_grid[new_growing] = CELL_GROWING
        self.grow_timer[new_growing] = 0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_cell(self, r: int, c: int) -> tuple[int, int, int]:
        val = self.resource_grid[r, c]
        if val == CELL_RIPE:
            return COLOR_RIPE
        elif val == CELL_GROWING:
            return COLOR_GROWING
        return COLOR_BG

    def _get_agent_color(self, agent: str) -> tuple[int, int, int]:
        idx = self.possible_agents.index(agent)
        return AGENT_PALETTE[idx % len(AGENT_PALETTE)]


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = CommonsHarvestEnv(n_agents=4, grid_size=15, max_cycles=50, render_mode="rgb_array")
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
    print("commons_harvest_v0: smoke test PASSED")
