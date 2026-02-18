"""Entropy-gradient-based action selection for DigiSoup agents.

v9 adds resource conservation (sustainable harvesting for Commons):
- When in a dense resource patch that is actively depleting, back off
- Perception-driven: uses growth_rate signal (dS/dt < 0 = patch dying)
- Survival overrides conservation (Rule 2 still fires when energy low)

v8 upgrades perception with thermodynamic sensing:
- 4x4 fine-grained entropy gradient (replaces 2x2 quadrants)
- Entropy growth gradient: follow where entropy is INCREASING (apple regrowth)
- KL divergence anomaly: find agents in dark environments

v4 added spatial memory (slime mold path reinforcement).
v3 added temporal phase cycling (jellyfish-inspired oscillation).

Priority rules (modified by phase + memory + growth + anomaly + conservation):
1. Random exploration — higher in explore phase, lower in exploit
2. Energy critically low -> seek resources or follow memory or growth
2.5. Resource conservation: dense patch + depletion -> back off
3. Exploit phase: seek resources at moderate energy (memory/growth-assisted)
4. Agents nearby (colour OR anomaly) -> phase-dependent cooperation threshold
5. Stable environment -> follow growth gradient or entropy gradient
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
MEMORY_FOLLOW_RECENCY = 0.1  # follow memory only if recency above this

# KL anomaly detection (agents in dark environments)
ANOMALY_AGENT_THRESHOLD = 0.3  # KL above this = likely an agent in dark arena

# Resource conservation (sustainable harvesting for commons)
CONSERVATION_DENSITY = 0.02    # resource density above this = "dense patch"
CONSERVATION_DEPLETION = -0.1  # growth_rate below this = "actively depleting"


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


def _has_growth(perception: 'Perception') -> bool:
    """Check if there's a usable entropy growth signal."""
    return np.linalg.norm(perception.growth_gradient) > 0.05


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
    # Cascade: visible resources > memory > growth gradient > entropy gradient.
    if state.energy < LOW_ENERGY_THRESHOLD:
        if perception.resources_nearby:
            return _move_toward(perception.resource_direction, rng)
        elif _has_memory(state):
            return _move_toward(state.resource_memory, rng)
        elif _has_growth(perception):
            return _move_toward(perception.growth_gradient, rng)
        else:
            return _move_toward(perception.gradient, rng)

    # Rule 2.5: Resource conservation — back off from depleting patches.
    # In Commons Harvest, apple regrowth requires neighboring apples. If the
    # agent strip-mines a patch, regrowth stops entirely (tipping point). When
    # we detect dense resources AND negative growth rate (entropy declining =
    # patch being depleted), move away to let it regrow. Survival (Rule 2)
    # overrides this — a starving agent still eats.
    if (perception.resources_nearby
            and perception.resource_density > CONSERVATION_DENSITY
            and perception.growth_rate < CONSERVATION_DEPLETION):
        return _move_away(perception.resource_direction, rng)

    # Rule 3: Exploit phase bonus — seek resources at moderate energy.
    # Growth gradient added as fallback after memory.
    if phase == "exploit" and state.energy < EXPLOIT_ENERGY_SEEK:
        if perception.resources_nearby:
            return _move_toward(perception.resource_direction, rng)
        elif _has_memory(state):
            return _move_toward(state.resource_memory, rng)
        elif _has_growth(perception):
            return _move_toward(perception.growth_gradient, rng)

    # Rule 4: Agents nearby — phase-dependent cooperation threshold.
    # Now also detects agents via KL anomaly in dark environments.
    # Explore: only interact if strongly cooperative (threshold 0.7).
    # Exploit: interact more readily (threshold 0.3).
    agents_detected = perception.agents_nearby
    agent_dir = perception.agent_direction
    if not agents_detected and perception.anomaly_strength > ANOMALY_AGENT_THRESHOLD:
        # KL anomaly detected — likely an agent in a dark arena
        agents_detected = True
        agent_dir = perception.anomaly_direction

    if agents_detected:
        coop_threshold = 0.7 if phase == "explore" else 0.3
        if state.cooperation_tendency > coop_threshold:
            return interact_action
        else:
            return _move_away(agent_dir, rng)

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
        # Growth gradient: prefer moving toward where entropy is increasing
        if _has_growth(perception):
            return _move_toward(perception.growth_gradient, rng)
        return _move_toward(perception.gradient, rng)
    else:
        # Chaotic: exploit current role strategy.
        return _exploit_role(state, perception, n_actions, rng)
