"""
Shared grid-world base class for DigiSoup substrate environments.

Provides common rendering, movement, observation windowing, and AEC cycle
management so that individual substrates only need to implement their
game-specific resource/interaction logic.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector


# ---------------------------------------------------------------------------
# Sprite / colour constants
# ---------------------------------------------------------------------------
CELL_PX = 8                       # 8x8 pixel sprites per grid cell
OBS_RADIUS = 5                    # 5 cells in every direction -> 11x11 view
OBS_CELLS = 2 * OBS_RADIUS + 1   # 11
OBS_SIZE = OBS_CELLS * CELL_PX    # 88  (88x88 px observation)

# Default agent colour palette (up to 16 agents)
AGENT_PALETTE = [
    (66, 135, 245),    # blue
    (245, 166, 35),    # orange
    (126, 211, 33),    # lime
    (208, 2, 27),      # red
    (189, 16, 224),    # purple
    (80, 227, 194),    # teal
    (255, 220, 0),     # yellow
    (255, 105, 180),   # pink
    (100, 100, 100),   # gray
    (0, 128, 128),     # dark teal
    (210, 105, 30),    # chocolate
    (148, 103, 189),   # muted purple
    (255, 127, 80),    # coral
    (0, 191, 255),     # deep sky blue
    (154, 205, 50),    # yellow-green
    (219, 112, 147),   # pale violet red
]

# Action indices
ACTION_NOOP       = 0
ACTION_UP         = 1
ACTION_DOWN       = 2
ACTION_LEFT       = 3
ACTION_RIGHT      = 4
ACTION_TURN_LEFT  = 5
ACTION_TURN_RIGHT = 6
ACTION_INTERACT   = 7
NUM_ACTIONS       = 8

# Direction vectors: 0=N, 1=E, 2=S, 3=W  (row, col deltas)
DIR_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class GridWorldBaseEnv(AECEnv):
    """Abstract base for 2-D grid-world AEC environments.

    Subclasses must implement:
        _reset_world()          -- populate grid, place agents, set resources
        _process_interaction()  -- handle the INTERACT action per agent
        _tick_world()           -- called once per full cycle (after all agents step)
        _render_cell(r, c)      -- return (R, G, B) for that grid cell
        _get_agent_color(agent) -- return (R, G, B) for agent's sprite
    """

    metadata: dict[str, Any] = {
        "render_modes": ["human", "rgb_array"],
        "name": "grid_world_base_v0",
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        n_agents: int = 7,
        grid_rows: int = 25,
        grid_cols: int = 25,
        max_cycles: int = 1000,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_cycles = max_cycles
        self.render_mode = render_mode

        self.possible_agents = [f"player_{i}" for i in range(n_agents)]

        # Spaces -- constant across agents
        self._obs_space = spaces.Box(
            low=0, high=255, shape=(OBS_SIZE, OBS_SIZE, 3), dtype=np.uint8
        )
        self._action_space = spaces.Discrete(NUM_ACTIONS)

        # These are set/reset in reset()
        self.agents: list[str] = []
        self.agent_positions: dict[str, tuple[int, int]] = {}
        self.agent_directions: dict[str, int] = {}  # 0=N,1=E,2=S,3=W

        self._agent_selector: AgentSelector | None = None
        self.agent_selection: str = ""

        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}

        self._step_count = 0
        self._cycle_count = 0
        self._steps_in_cycle = 0

        # Episode stats
        self.episode_rewards: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Space helpers
    # ------------------------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return self._action_space

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        self.agents = self.possible_agents[:]
        self._agent_selector = AgentSelector(self.agents[:])
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._step_count = 0
        self._cycle_count = 0
        self._steps_in_cycle = 0

        self.episode_rewards = {a: 0.0 for a in self.agents}

        # Place agents randomly (subclass may override via _reset_world)
        self.agent_positions = {}
        self.agent_directions = {}

        # Subclass populates the grid, resources, and initial agent positions
        self._reset_world()

        # If subclass didn't set positions, scatter randomly
        if not self.agent_positions:
            self._scatter_agents()

    def _scatter_agents(self) -> None:
        """Place agents on random empty cells."""
        occupied: set[tuple[int, int]] = set()
        for agent in self.agents:
            while True:
                r = self._rng.integers(0, self.grid_rows)
                c = self._rng.integers(0, self.grid_cols)
                if (r, c) not in occupied and self._cell_walkable(r, c):
                    self.agent_positions[agent] = (r, c)
                    self.agent_directions[agent] = int(self._rng.integers(0, 4))
                    occupied.add((r, c))
                    break

    def _cell_walkable(self, r: int, c: int) -> bool:  # noqa: ARG002
        """Override to block certain cells from agent placement."""
        return True

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: int) -> None:
        agent = self.agent_selection

        # Dead-agent bookkeeping
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        # Clear this agent's reward from prior step
        self.rewards[agent] = 0.0

        # Execute action
        self._execute_action(agent, action)

        # Track steps
        self._step_count += 1
        self._steps_in_cycle += 1

        # Accumulate rewards
        self._accumulate_rewards()

        # Episode reward tracking
        self.episode_rewards[agent] += self.rewards[agent]

        # If all agents have acted this cycle, tick the world
        if self._agent_selector.is_last():
            self._cycle_count += 1
            self._steps_in_cycle = 0
            self._tick_world()

            # Check truncation
            if self._cycle_count >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True

        # Advance to next agent
        self.agent_selection = self._agent_selector.next()

    def _execute_action(self, agent: str, action: int) -> None:
        r, c = self.agent_positions[agent]
        d = self.agent_directions[agent]

        if action == ACTION_NOOP:
            pass
        elif action in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            dr, dc = {
                ACTION_UP:    (-1, 0),
                ACTION_DOWN:  (1, 0),
                ACTION_LEFT:  (0, -1),
                ACTION_RIGHT: (0, 1),
            }[action]
            nr, nc = r + dr, c + dc
            if self._in_bounds(nr, nc) and self._cell_walkable(nr, nc):
                # Check no other agent occupies that cell
                if not self._cell_occupied_by_other(agent, nr, nc):
                    self.agent_positions[agent] = (nr, nc)
                    self._on_agent_move(agent, nr, nc)
        elif action == ACTION_TURN_LEFT:
            self.agent_directions[agent] = (d - 1) % 4
        elif action == ACTION_TURN_RIGHT:
            self.agent_directions[agent] = (d + 1) % 4
        elif action == ACTION_INTERACT:
            self._process_interaction(agent)

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.grid_rows and 0 <= c < self.grid_cols

    def _cell_occupied_by_other(self, agent: str, r: int, c: int) -> bool:
        for a, pos in self.agent_positions.items():
            if a != agent and pos == (r, c):
                return True
        return False

    def _on_agent_move(self, agent: str, r: int, c: int) -> None:
        """Called after an agent moves to (r, c). Override for harvesting."""
        pass

    def _was_dead_step(self, action: int) -> None:
        """Handle step for terminated/truncated agents (PettingZoo convention).

        After all agents have had their dead step processed, remove them so
        the ``while env.agents`` loop terminates.
        """
        agent = self.agent_selection

        # Remove this agent from the active list
        if agent in self.agents:
            self.agents.remove(agent)

        # If no agents left, we're done
        if not self.agents:
            return

        # Rebuild selector with remaining agents
        self._agent_selector = AgentSelector(self.agents[:])
        self.agent_selection = self._agent_selector.reset()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def observe(self, agent: str) -> np.ndarray:
        """Return an 88x88x3 RGB observation centred on the agent."""
        ar, ac = self.agent_positions[agent]
        obs = np.zeros((OBS_SIZE, OBS_SIZE, 3), dtype=np.uint8)

        for dr in range(-OBS_RADIUS, OBS_RADIUS + 1):
            for dc in range(-OBS_RADIUS, OBS_RADIUS + 1):
                wr, wc = ar + dr, ac + dc
                cell_r = (dr + OBS_RADIUS) * CELL_PX
                cell_c = (dc + OBS_RADIUS) * CELL_PX

                if self._in_bounds(wr, wc):
                    color = self._render_cell(wr, wc)
                else:
                    color = (0, 0, 0)  # out of bounds = black

                obs[cell_r:cell_r + CELL_PX, cell_c:cell_c + CELL_PX] = color

        # Paint agents that fall within the observation window
        for a in self.agents:
            if self.terminations.get(a, False):
                continue
            pr, pc = self.agent_positions[a]
            dr, dc = pr - ar, pc - ac
            if abs(dr) <= OBS_RADIUS and abs(dc) <= OBS_RADIUS:
                cell_r = (dr + OBS_RADIUS) * CELL_PX
                cell_c = (dc + OBS_RADIUS) * CELL_PX
                agent_color = self._get_agent_color(a)
                # Paint agent as a 6x6 block centred in the 8x8 cell
                obs[cell_r + 1:cell_r + 7, cell_c + 1:cell_c + 7] = agent_color

        # Highlight self with a 1-pixel white border
        center_r = OBS_RADIUS * CELL_PX
        center_c = OBS_RADIUS * CELL_PX
        obs[center_r:center_r + CELL_PX, center_c, :] = 255
        obs[center_r:center_r + CELL_PX, center_c + CELL_PX - 1, :] = 255
        obs[center_r, center_c:center_c + CELL_PX, :] = 255
        obs[center_r + CELL_PX - 1, center_c:center_c + CELL_PX, :] = 255

        return obs

    # ------------------------------------------------------------------
    # Render (full world view)
    # ------------------------------------------------------------------
    def render(self) -> np.ndarray | None:
        img = np.zeros((self.grid_rows * CELL_PX, self.grid_cols * CELL_PX, 3),
                        dtype=np.uint8)

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                color = self._render_cell(r, c)
                pr, pc = r * CELL_PX, c * CELL_PX
                img[pr:pr + CELL_PX, pc:pc + CELL_PX] = color

        for a in self.agents:
            if self.terminations.get(a, False):
                continue
            ar, ac = self.agent_positions[a]
            pr, pc = ar * CELL_PX, ac * CELL_PX
            agent_color = self._get_agent_color(a)
            img[pr + 1:pr + 7, pc + 1:pc + 7] = agent_color

        if self.render_mode == "rgb_array":
            return img
        return img

    # ------------------------------------------------------------------
    # State (global observation for centralized training)
    # ------------------------------------------------------------------
    def state(self) -> np.ndarray:
        return self.render()

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------
    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Subclass hooks (must be implemented)
    # ------------------------------------------------------------------
    def _reset_world(self) -> None:
        raise NotImplementedError

    def _process_interaction(self, agent: str) -> None:
        raise NotImplementedError

    def _tick_world(self) -> None:
        raise NotImplementedError

    def _render_cell(self, r: int, c: int) -> tuple[int, int, int]:
        raise NotImplementedError

    def _get_agent_color(self, agent: str) -> tuple[int, int, int]:
        idx = self.possible_agents.index(agent)
        return AGENT_PALETTE[idx % len(AGENT_PALETTE)]
