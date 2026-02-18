"""Entropy-gradient-based action selection for DigiSoup agents.

v5 adds adaptive cooperation threshold (vampire bat reciprocity):
The cooperation threshold adjusts based on recent interaction success.
If interactions have been working (entropy changed), lower the bar —
cooperate more readily. If they've been failing, raise it — be reluctant.
Uses the existing interaction_outcomes history, no new state needed.

v4 added spatial memory (slime mold path reinforcement).
v3 added temporal phase cycling (jellyfish-inspired oscillation).

Priority rules (modified by phase + memory + adaptive cooperation):
1. Random exploration — higher in explore phase, lower in exploit
2. Energy critically low -> seek resources or follow memory
3. Exploit phase: seek resources at moderate energy (memory-assisted)
4. Agents nearby -> adaptive cooperation threshold
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
    get_interaction_success_rate,
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

# Adaptive cooperation thresholds (vampire bat reciprocity)
COOP_BASE_EXPLORE = 0.7   # base threshold in explore phase
COOP_BASE_EXPLOIT = 0.3   # base threshold in exploit phase
COOP_ADAPT_RANGE = 0.2    # max adjustment up or down from base


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

def _adaptive_coop_threshold(state: DigiSoupState, phase: str) -> float:
    """Compute cooperation threshold adapted by recent interaction success.

    Vampire bat reciprocity: if interactions have been working, lower the
    threshold (cooperate more readily). If failing, raise it (be reluctant).
    With insufficient data, use the base phase-dependent threshold.
    """
    base = COOP_BASE_EXPLORE if phase == "explore" else COOP_BASE_EXPLOIT
    success_rate = get_interaction_success_rate(state)
    if success_rate is None:
        return base
    # success_rate 0.0 -> raise threshold by ADAPT_RANGE (reluctant)
    # success_rate 0.5 -> no change
    # success_rate 1.0 -> lower threshold by ADAPT_RANGE (eager)
    adjustment = COOP_ADAPT_RANGE * (success_rate - 0.5) * 2.0
    return max(0.05, min(0.95, base - adjustment))


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

    # Rule 4: Agents nearby — adaptive cooperation threshold.
    # Base thresholds (explore: 0.7, exploit: 0.3) adjusted by recent
    # interaction success rate. Vampire bat: reciprocity drives cooperation.
    if perception.agents_nearby:
        coop_threshold = _adaptive_coop_threshold(state, phase)
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
