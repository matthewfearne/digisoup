"""Entropy-gradient-based action selection for DigiSoup agents.

Action selection follows 6 priority rules:
1. Compute observation entropy and gradient direction
2. Energy low -> move toward resources
3. Agents nearby -> cooperation tendency determines interact vs avoid
4. Environment stable -> explore (move toward higher entropy)
5. Environment chaotic -> exploit current strategy
6. Periodic random exploration proportional to internal entropy

NO reward optimization. NO training. Every rule explainable in one paragraph.

Melting Pot action space:
  0: no-op  1: forward  2: backward  3: left
  4: right  5: turn left  6: turn right  7: interact
"""
from __future__ import annotations

import numpy as np

from .perception import Perception, MAX_ENTROPY
from .state import DigiSoupState, LOW_ENERGY_THRESHOLD, get_role


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


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

STABLE_THRESHOLD = 0.3    # change below this = "stable" environment
COOP_THRESHOLD = 0.5      # cooperation tendency above this = interact


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
    """Choose action using entropy-gradient priority rules.

    The six rules implement DigiSoup's core insight: agents don't need
    reward to behave intelligently. Entropy gradients tell you where
    interesting things are. Internal energy tells you when to forage.
    Cooperation tendency tells you how to respond to other agents.
    """
    rng = rng or np.random.default_rng()
    interact_action = n_actions - 1

    # Rule 6 (checked first): Random exploration proportional to entropy.
    # Higher entropy in the environment = more exploration.
    explore_prob = 0.05 + 0.15 * min(perception.entropy / MAX_ENTROPY, 1.0)
    if rng.random() < explore_prob:
        return int(rng.integers(0, n_actions))

    # Rule 2: Energy low -> move toward resources.
    # When hungry, prioritise finding food over everything else.
    if state.energy < LOW_ENERGY_THRESHOLD:
        if perception.resources_nearby:
            return _move_toward(perception.resource_direction, rng)
        else:
            # No resources visible, explore to find some
            return _move_toward(perception.gradient, rng)

    # Rule 3: Agents nearby -> cooperation tendency determines behavior.
    # High cooperation tendency = interact. Low = avoid.
    if perception.agents_nearby:
        if state.cooperation_tendency > COOP_THRESHOLD:
            return interact_action
        else:
            return _move_away(perception.agent_direction, rng)

    # Rules 4 & 5: Environment stability determines explore vs exploit.
    if perception.change < STABLE_THRESHOLD:
        # Rule 4: Stable environment -> explore (seek higher entropy).
        return _move_toward(perception.gradient, rng)
    else:
        # Rule 5: Chaotic environment -> exploit current strategy.
        return _exploit_role(state, perception, n_actions, rng)
