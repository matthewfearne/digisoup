"""DigiSoup Policy -- official Melting Pot Policy interface.

v14 "Hive Mind": shared spatial memory between all focal agents.
When one agent finds dirt/resources, all agents learn the location.
Mycorrhizal network — one node's discovery benefits the whole colony.

NO neural networks. NO reward optimization. NO training.
"""
from __future__ import annotations

import dm_env
import numpy as np

from .perception import perceive
from .state import DigiSoupState, initial_state, update_state, ego_to_world
from .action import select_action


# ---------------------------------------------------------------------------
# Hive Memory — shared across all focal agent instances
# ---------------------------------------------------------------------------

class HiveMemory:
    """Shared spatial memory between all DigiSoup agent instances.

    When any agent detects dirt or resources, it writes the estimated
    world position to the hive. Other agents can query the hive for
    the nearest point of interest and compute a direction toward it.

    Thread-safe is not needed: Melting Pot runs agents sequentially.
    """

    def __init__(self) -> None:
        self.points: list[tuple[float, float, str]] = []  # (y, x, type)
        self._seen: set[tuple[int, int, str]] = set()     # dedup grid

    def report(self, world_y: float, world_x: float, kind: str) -> None:
        """Agent reports a discovery at estimated world position."""
        # Quantize to grid to avoid duplicate spam
        key = (int(round(world_y)), int(round(world_x)), kind)
        if key not in self._seen:
            self._seen.add(key)
            self.points.append((world_y, world_x, kind))

    def query(self, agent_y: float, agent_x: float,
              max_dist: float = 50.0) -> tuple[bool, np.ndarray]:
        """Get direction from agent's position toward nearest hive point.

        Returns (has_signal, world_direction_unit_vector).
        """
        if not self.points:
            return False, np.zeros(2)

        best_dist = max_dist
        best_dir = np.zeros(2)
        for py, px, _ in self.points:
            dy = py - agent_y
            dx = px - agent_x
            dist = (dy**2 + dx**2) ** 0.5
            if 1.0 < dist < best_dist:  # skip if already there
                best_dist = dist
                best_dir = np.array([dy, dx])

        norm = np.linalg.norm(best_dir)
        if norm < 1e-6:
            return False, np.zeros(2)
        return True, best_dir / norm

    def reset(self) -> None:
        """Clear all shared memory (call between episodes)."""
        self.points.clear()
        self._seen.clear()


class DigiSoupPolicy:
    """Zero-training entropy-driven policy for Melting Pot.

    Implements the same interface as meltingpot's Policy ABC.
    The RNG lives on the policy object for efficiency. All biological
    state (energy, cooperation, role) flows through DigiSoupState.

    Hive memory is class-level: all instances share spatial discoveries.
    """

    _hive: HiveMemory | None = None  # shared across all instances

    @classmethod
    def _get_hive(cls) -> HiveMemory:
        if cls._hive is None:
            cls._hive = HiveMemory()
        return cls._hive

    @classmethod
    def reset_hive(cls) -> None:
        """Reset shared memory between episodes."""
        if cls._hive is not None:
            cls._hive.reset()

    def __init__(self, seed: int = 42, n_actions: int = 8) -> None:
        self._rng = np.random.default_rng(seed)
        self._n_actions = n_actions
        self._hive_ref = self._get_hive()

    def initial_state(self) -> DigiSoupState:
        """Return the initial agent state. Resets hive for new episode."""
        self.reset_hive()
        return initial_state()

    def step(
        self, timestep: dm_env.TimeStep, prev_state: DigiSoupState
    ) -> tuple[int, DigiSoupState]:
        """Observe -> Perceive -> Hive write -> Act -> Update state.

        Reward in timestep is RECORDED but NEVER used for action selection.
        """
        # Extract RGB observation
        obs = timestep.observation
        if hasattr(obs, "get"):
            obs = obs.get("RGB", np.zeros((88, 88, 3), dtype=np.uint8))
        obs = np.asarray(obs, dtype=np.uint8)

        # Perceive: entropy, gradients, growth, anomaly, agent/resource, change
        prev_obs = prev_state.prev_obs if prev_state.has_prev_obs else None
        prev_grid = prev_state.prev_entropy_grid if prev_state.has_prev_obs else None
        perception = perceive(obs, prev_obs, prev_grid)

        # Hive write: share discoveries with all agents
        hive = self._hive_ref
        if perception.dirt_nearby:
            # Estimate world position of dirt: agent pos + ~5 tiles in dirt direction
            world_dir = ego_to_world(perception.dirt_direction, prev_state.orientation)
            dirt_pos = prev_state.position + 5.0 * world_dir
            hive.report(dirt_pos[0], dirt_pos[1], "dirt")
        if perception.resources_nearby:
            world_dir = ego_to_world(perception.resource_direction, prev_state.orientation)
            res_pos = prev_state.position + 5.0 * world_dir
            hive.report(res_pos[0], res_pos[1], "resource")

        # Hive read: get direction toward nearest shared discovery
        has_hive, hive_world_dir = hive.query(
            prev_state.position[0], prev_state.position[1]
        )

        # Select action using perception + internal state + hive signal
        action = select_action(
            perception, prev_state, self._n_actions, self._rng,
            hive_direction=hive_world_dir if has_hive else None,
        )

        # Update internal state (energy, cooperation, entropy estimate, memory)
        new_state = update_state(
            prev_state, obs, action,
            perception.entropy, perception.change,
            resources_nearby=perception.resources_nearby,
            resource_direction=perception.resource_direction,
            resource_density=perception.resource_density,
            entropy_grid=perception.entropy_grid,
        )

        return int(action), new_state

    def close(self) -> None:
        """Clean up resources. Nothing to clean for DigiSoup."""
        pass
