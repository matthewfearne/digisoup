"""DigiSoup Policy -- official Melting Pot Policy interface.

This is v1: pure random baseline. The entropy layers get added
incrementally in subsequent versions.

The Policy interface requires:
    initial_state() -> State
    step(timestep, prev_state) -> (action, next_state)
    close() -> None

State must be immutable and contain all mutable agent state.
The policy itself must have no mutable state outside of what's in State.

Observation permitted keys (from Melting Pot evaluation):
    RGB (88, 88, 3) uint8 -- the only one we use
    READY_TO_SHOOT, INVENTORY, HUNGER, STAMINA, COLLECTIVE_REWARD
    (POSITION and ORIENTATION are NOT permitted during evaluation)
"""
from __future__ import annotations

from typing import Any, NamedTuple

import dm_env
import numpy as np

from agents.digisoup.entropy import compute_entropy_state, EntropyState, EMPTY_ENTROPY_STATE
from agents.digisoup.action import select_action


class AgentState(NamedTuple):
    """Immutable agent state passed between steps."""
    step_count: int
    entropy_state: EntropyState
    rng_key: int


class DigiSoupPolicy:
    """Zero-training entropy-driven policy for Melting Pot.

    Implements the same interface as meltingpot's Policy ABC.
    No mutable state on the policy object -- all state flows through AgentState.
    """

    def __init__(self, seed: int = 42, n_actions: int = 8) -> None:
        self._seed = seed
        self._n_actions = n_actions

    def initial_state(self) -> AgentState:
        """Return the initial agent state. No side effects."""
        return AgentState(
            step_count=0,
            entropy_state=EMPTY_ENTROPY_STATE,
            rng_key=self._seed,
        )

    def step(
        self, timestep: dm_env.TimeStep, prev_state: AgentState
    ) -> tuple[int, AgentState]:
        """Select an action from the observation. No side effects.

        Reward in timestep is RECORDED but NEVER used for action selection.
        """
        # Extract RGB observation
        obs = timestep.observation
        if isinstance(obs, dict):
            rgb = obs.get("RGB", np.zeros((88, 88, 3), dtype=np.uint8))
        else:
            rgb = np.zeros((88, 88, 3), dtype=np.uint8)
        rgb = np.asarray(rgb, dtype=np.uint8)

        # Advance RNG deterministically
        rng = np.random.default_rng(prev_state.rng_key)
        next_rng_key = int(rng.integers(0, 2**31))

        # Compute updated entropy state from observation
        new_entropy_state = compute_entropy_state(
            rgb, prev_state.entropy_state
        )

        # Select action using entropy state
        action = select_action(
            entropy_state=new_entropy_state,
            n_actions=self._n_actions,
            step_count=prev_state.step_count,
            rng=rng,
        )

        new_state = AgentState(
            step_count=prev_state.step_count + 1,
            entropy_state=new_entropy_state,
            rng_key=next_rng_key,
        )

        return int(action), new_state

    def close(self) -> None:
        """Clean up resources. Nothing to clean for DigiSoup."""
        pass
