"""Entropy-gradient-based action selection for DigiSoup agents.

v3 adds temporal phase cycling (jellyfish-inspired oscillation):
The agent alternates between EXPLORE and EXPLOIT phases on a fixed
clock. This creates structured behavior — discover first, then act.

Priority rules (modified by phase):
1. Random exploration — higher in explore phase, lower in exploit
2. Energy critically low -> seek resources (always)
3. Exploit phase: seek resources at moderate energy too
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
from .state import DigiSoupState, LOW_ENERGY_THRESHOLD, get_role, get_phase


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

def select_action(
    perception: Perception,
    state: DigiSoupState,
    n_actions: int = 8,
    rng: np.random.Generator | None = None,
) -> int:
    """Choose action using phase-modulated entropy-gradient priority rules.

    The agent alternates between explore and exploit phases on a fixed
    clock. Explore phase: higher exploration, scanning turns, reluctant
    to interact. Exploit phase: lower exploration, eager to interact
    and gather resources.
    """
    rng = rng or np.random.default_rng()
    interact_action = n_actions - 1
    phase = get_phase(state)

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
    if state.energy < LOW_ENERGY_THRESHOLD:
        if perception.resources_nearby:
            return _move_toward(perception.resource_direction, rng)
        else:
            return _move_toward(perception.gradient, rng)

    # Rule 3: Exploit phase bonus — seek resources at moderate energy.
    # Don't wait until starving; gather while the getting is good.
    if phase == "exploit" and state.energy < EXPLOIT_ENERGY_SEEK:
        if perception.resources_nearby:
            return _move_toward(perception.resource_direction, rng)

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
        return _move_toward(perception.gradient, rng)
    else:
        # Chaotic: exploit current role strategy.
        return _exploit_role(state, perception, n_actions, rng)
