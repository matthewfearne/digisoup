"""Entropy-gradient-based action selection for DigiSoup agents.

v6 adds entropy-as-energy with cockroach persistence:
Energy depletion scales with environmental entropy — rich environments sustain,
barren ones drain. Below COCKROACH_THRESHOLD, the agent enters survival mode:
no random exploration, no cooperation, pure resource-seeking. Cockroach
persistence: the last organism standing in any hostile environment.

v4 added spatial memory (slime mold path reinforcement).
v3 added temporal phase cycling (jellyfish-inspired oscillation).

Priority rules (modified by phase + memory + cockroach mode):
0. Cockroach mode -> pure resource-seeking (overrides all other rules)
1. Random exploration — higher in explore phase, lower in exploit
2. Energy critically low -> seek resources or follow memory
3. Exploit phase: seek resources at moderate energy (memory-assisted)
4. Agents nearby -> phase-dependent cooperation threshold
5. Stable environment -> explore gradient (with scan bias in explore phase)
6. Chaotic environment -> exploit current role

NO reward optimization. NO training. Every rule explainable in one paragraph.

Melting Pot action space:
  0: no-op  1: forward  2: backward  3: left
  4: right  5: turn left  6: turn right  7: interact
"""
from __future__ import annotations

import numpy as np

from .perception import Perception, MAX_ENTROPY
from .state import (
    DigiSoupState, LOW_ENERGY_THRESHOLD, get_role, get_phase,
    is_cockroach_mode,
)


# ---------------------------------------------------------------------------
# Action constants (Melting Pot standard 8-action space)
# ---------------------------------------------------------------------------

NOOP = 0
FORWARD = 1
BACKWARD = 2
LEFT = 3
RIGHT = 4
TURN_LEFT = 5
TURN_RIGHT = 6
INTERACT = 7

# Movement actions for explore-phase random selection
_MOVE_ACTIONS = [FORWARD, BACKWARD, LEFT, RIGHT, TURN_LEFT, TURN_RIGHT]


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

STABLE_THRESHOLD = 0.3    # change below this = "stable" environment
EXPLOIT_ENERGY_SEEK = 0.7 # in exploit phase, seek resources below this energy
MEMORY_FOLLOW_RECENCY = 0.1  # follow memory only if recency above this


# ---------------------------------------------------------------------------
# Direction-to-action helpers
# ---------------------------------------------------------------------------

def _move_toward(direction: np.ndarray, rng: np.random.Generator) -> int:
    """Convert a (dy, dx) direction vector to a movement action.

    Adds small noise for natural-looking movement. dy<0 = upward
    in image space, which maps to FORWARD (toward top of view).
    """
    noisy = direction + rng.normal(0, 0.2, size=2)
    norm = np.linalg.norm(noisy)

    if norm < 0.05:
        return FORWARD

    dy, dx = float(noisy[0]), float(noisy[1])

    if abs(dy) >= abs(dx):
        return FORWARD if dy < 0 else BACKWARD
    else:
        return RIGHT if dx > 0 else LEFT


def _move_away(direction: np.ndarray, rng: np.random.Generator) -> int:
    """Move in the opposite direction."""
    return _move_toward(-direction, rng)


def _exploit_role(
    state: DigiSoupState,
    perception: Perception,
    n_actions: int,
    rng: np.random.Generator,
) -> int:
    """Exploit current strategy based on emergent role.

    When the environment is chaotic (lots of change), stick with what
    you've been doing. Cooperators keep interacting. Explorers keep
    moving toward change. Scanners keep turning. Generalists follow
    the entropy gradient.
    """
    role = get_role(state)

    if role == "cooperator":
        return n_actions - 1  # INTERACT
    elif role == "explorer":
        return _move_toward(perception.change_direction, rng)
    elif role == "scanner":
        return TURN_LEFT if rng.random() < 0.5 else TURN_RIGHT
    else:
        return _move_toward(perception.gradient, rng)


# ---------------------------------------------------------------------------
# Main action selection
# ---------------------------------------------------------------------------

def _has_memory(state: DigiSoupState) -> bool:
    """Check if the agent has a usable resource memory."""
    return (
        state.resource_recency > MEMORY_FOLLOW_RECENCY
        and np.linalg.norm(state.resource_memory) > 0.01
    )


def select_action(
    perception: Perception,
    state: DigiSoupState,
    n_actions: int = 8,
    rng: np.random.Generator | None = None,
) -> int:
    """Choose action using phase-modulated entropy-gradient priority rules.

    The agent alternates between explore and exploit phases on a fixed
    clock. Spatial memory (slime mold path reinforcement) biases movement
    toward remembered resource locations when nothing is currently visible.
    """
    rng = rng or np.random.default_rng()
    interact_action = n_actions - 1
    phase = get_phase(state)

    # Rule 0: Cockroach survival mode — overrides everything.
    # At critically low energy, no exploration, no cooperation, pure survival.
    # Seek resources by any means: visible > memory > entropy gradient.
    if is_cockroach_mode(state):
        if perception.resources_nearby:
            return _move_toward(perception.resource_direction, rng)
        elif _has_memory(state):
            return _move_toward(state.resource_memory, rng)
        else:
            # Desperate: move toward highest entropy (most likely to have stuff)
            return _move_toward(perception.gradient, rng)

    # Rule 1: Random exploration — phase modulates probability.
    # Explore phase: cast a wider net. Exploit phase: stay focused.
    entropy_ratio = min(perception.entropy / MAX_ENTROPY, 1.0)
    if phase == "explore":
        explore_prob = 0.10 + 0.20 * entropy_ratio
    else:
        explore_prob = 0.02 + 0.08 * entropy_ratio

    if rng.random() < explore_prob:
        if phase == "explore":
            # Bias random toward movement and scanning (not interact/noop)
            return int(rng.choice(_MOVE_ACTIONS))
        return int(rng.integers(0, n_actions))

    # Rule 2: Energy critically low -> seek resources (always priority).
    # Memory-assisted: if no resources visible, follow scent trail.
    if state.energy < LOW_ENERGY_THRESHOLD:
        if perception.resources_nearby:
            return _move_toward(perception.resource_direction, rng)
        elif _has_memory(state):
            return _move_toward(state.resource_memory, rng)
        else:
            return _move_toward(perception.gradient, rng)

    # Rule 3: Exploit phase bonus — seek resources at moderate energy.
    # Memory-assisted: follow trail even when resources not visible.
    if phase == "exploit" and state.energy < EXPLOIT_ENERGY_SEEK:
        if perception.resources_nearby:
            return _move_toward(perception.resource_direction, rng)
        elif _has_memory(state):
            return _move_toward(state.resource_memory, rng)

    # Rule 4: Agents nearby — phase-dependent cooperation threshold.
    # Explore: only interact if strongly cooperative (threshold 0.7).
    # Exploit: interact more readily (threshold 0.3).
    if perception.agents_nearby:
        coop_threshold = 0.7 if phase == "explore" else 0.3
        if state.cooperation_tendency > coop_threshold:
            return interact_action
        else:
            return _move_away(perception.agent_direction, rng)

    # Rules 5 & 6: Environment stability determines movement strategy.
    if perception.change < STABLE_THRESHOLD:
        # Stable: follow entropy gradient. Explore phase adds scan turns.
        if phase == "explore" and rng.random() < 0.3:
            return TURN_LEFT if rng.random() < 0.5 else TURN_RIGHT
        if phase == "exploit" and perception.resources_nearby:
            return _move_toward(perception.resource_direction, rng)
        # Memory fallback: if stable and no resources, follow the trail
        if phase == "exploit" and _has_memory(state):
            return _move_toward(state.resource_memory, rng)
        return _move_toward(perception.gradient, rng)
    else:
        # Chaotic: exploit current role strategy.
        return _exploit_role(state, perception, n_actions, rng)
