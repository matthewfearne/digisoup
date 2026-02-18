"""DigiSoup Policy -- official Melting Pot Policy interface.

Uses the original DigiSoup architecture:
- Perception: entropy, gradients, agent/resource detection, change detection
- Internal state: energy, cooperation tendency, role emergence
- Action: entropy-gradient priority rules

The Policy interface requires:
    initial_state() -> State
    step(timestep, prev_state) -> (action, next_state)
    close() -> None

NO neural networks. NO reward optimization. NO training.
"""
from __future__ import annotations

import dm_env
import numpy as np

from .perception import perceive
from .state import DigiSoupState, initial_state, update_state
from .action import select_action


class DigiSoupPolicy:
    """Zero-training entropy-driven policy for Melting Pot.

    Implements the same interface as meltingpot's Policy ABC.
    The RNG lives on the policy object for efficiency. All biological
    state (energy, cooperation, role) flows through DigiSoupState.
    """

    def __init__(self, seed: int = 42, n_actions: int = 8) -> None:
        self._rng = np.random.default_rng(seed)
        self._n_actions = n_actions

    def initial_state(self) -> DigiSoupState:
        """Return the initial agent state."""
        return initial_state()

    def step(
        self, timestep: dm_env.TimeStep, prev_state: DigiSoupState
    ) -> tuple[int, DigiSoupState]:
        """Observe -> Perceive -> Act -> Update state.

        Reward in timestep is RECORDED but NEVER used for action selection.
        """
        # Extract RGB observation
        obs = timestep.observation
        if hasattr(obs, "get"):
            obs = obs.get("RGB", np.zeros((88, 88, 3), dtype=np.uint8))
        obs = np.asarray(obs, dtype=np.uint8)

        # Perceive: entropy, gradients, agent/resource detection, change
        prev_obs = prev_state.prev_obs if prev_state.has_prev_obs else None
        perception = perceive(obs, prev_obs)

        # Select action using perception + internal state
        action = select_action(
            perception, prev_state, self._n_actions, self._rng
        )

        # Update internal state (energy, cooperation, entropy estimate, memory)
        new_state = update_state(
            prev_state, obs, action,
            perception.entropy, perception.change,
            resources_nearby=perception.resources_nearby,
            resource_direction=perception.resource_direction,
            resource_density=perception.resource_density,
        )

        return int(action), new_state

    def close(self) -> None:
        """Clean up resources. Nothing to clean for DigiSoup."""
        pass
