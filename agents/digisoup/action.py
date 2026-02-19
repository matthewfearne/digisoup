"""Entropy-gradient-based action selection for DigiSoup agents.

v12 "Echo Explorer":
- Starvation exploration: when no resources for 50+ steps AND memory/heatmap
  depleted, boost random exploration to 50%. Gets agent to the river in barren
  Clean Up scenarios (CU_1/4/5/6 zero-score fix).
- Echo feedback: successful INTERACT on dirt raises cooperation_tendency (already
  in state machine). Now Rule 2.5 also fires when cooperation is high (>0.6)
  even if resources are visible. Committed cleaners keep cleaning.

v11 base: cleaning rule. v10: colour fix, heatmap, heading. v8: 4x4 grid.

Priority rules:
1. Random exploration — phase-modulated, with starvation override
2. Energy critically low -> seek resources / clean dirt / memory / heatmap / growth
2.5. Dirt nearby + (no resources OR high cooperation) -> clean
3. Exploit phase: seek resources at moderate energy (memory/heatmap/growth-assisted)
4. Agents nearby (colour OR anomaly) -> phase-dependent cooperation threshold
5. Stable environment -> avoid crowds, follow growth or entropy gradient
6. Chaotic environment -> exploit current role

NO reward optimization. NO training. Every rule explainable in one paragraph.

Melting Pot action space:
  0: no-op  1: forward  2: backward  3: left
  4: right  5: turn left  6: turn right  7: interact
"""
from __future__ import annotations

import numpy as np

from .perception import Perception, MAX_ENTROPY, _grid_gradient
from .state import (
    DigiSoupState, LOW_ENERGY_THRESHOLD, get_role, get_phase,
    HEATMAP_THRESHOLD, HEADING_BLEND, HEADING_MIN_NORM,
    STARVATION_THRESHOLD,
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

# KL anomaly detection (agents in dark environments)
ANOMALY_AGENT_THRESHOLD = 0.3  # KL above this = likely an agent in dark arena

# Agent crowding avoidance threshold
CROWDING_THRESHOLD = 0.05      # agent_grid max above this = significant crowding

# Dirt cleaning (Clean Up substrate: pollution blocks apple growth)
DIRT_CLOSE_DENSITY = 0.01     # dirt density above this = close enough to INTERACT

# Starvation exploration (desperate wandering when all resource signals depleted)
STARVATION_EXPLORE_PROB = 0.50    # 50% random moves when starving

# Echo cleaning (INTERACT on dirt → coop rises → commit to cleaning)
ECHO_CLEANING_THRESHOLD = 0.6    # cooperation above this = committed cleaner


# ---------------------------------------------------------------------------
# Direction-to-action helpers
# ---------------------------------------------------------------------------

def _move_toward(
    direction: np.ndarray,
    rng: np.random.Generator,
    heading: np.ndarray | None = None,
) -> int:
    """Convert a (dy, dx) direction vector to a movement action.

    Adds small noise for natural-looking movement. dy<0 = upward
    in image space, which maps to FORWARD (toward top of view).
    If heading is provided, blends it for smoother directional persistence.
    """
    if heading is not None and np.linalg.norm(heading) > HEADING_MIN_NORM:
        effective = (1.0 - HEADING_BLEND) * direction + HEADING_BLEND * heading
    else:
        effective = direction
    noisy = effective + rng.normal(0, 0.2, size=2)
    norm = np.linalg.norm(noisy)

    if norm < 0.05:
        return FORWARD

    dy, dx = float(noisy[0]), float(noisy[1])

    if abs(dy) >= abs(dx):
        return FORWARD if dy < 0 else BACKWARD
    else:
        return RIGHT if dx > 0 else LEFT


def _move_away(
    direction: np.ndarray,
    rng: np.random.Generator,
    heading: np.ndarray | None = None,
) -> int:
    """Move in the opposite direction."""
    return _move_toward(-direction, rng, heading)


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


def _has_heatmap(state: DigiSoupState) -> bool:
    """Check if the resource heatmap has usable signal."""
    return float(state.resource_heatmap.max()) > HEATMAP_THRESHOLD


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
    Resource heatmap provides a broader spatial memory fallback.
    Heading persistence smooths movement trajectories.
    """
    rng = rng or np.random.default_rng()
    interact_action = n_actions - 1
    phase = get_phase(state)
    heading = state.heading

    # Rule 1: Random exploration — phase modulates probability.
    # Explore phase: cast a wider net. Exploit phase: stay focused.
    # Starvation override: when no resources for many steps AND all memory
    # depleted, dramatically boost exploration to escape barren areas.
    entropy_ratio = min(perception.entropy / MAX_ENTROPY, 1.0)
    is_starving = (
        state.starvation_steps > STARVATION_THRESHOLD
        and not _has_memory(state)
        and not _has_heatmap(state)
    )
    if is_starving:
        explore_prob = STARVATION_EXPLORE_PROB
    elif phase == "explore":
        explore_prob = 0.10 + 0.20 * entropy_ratio
    else:
        explore_prob = 0.02 + 0.08 * entropy_ratio

    if rng.random() < explore_prob:
        if phase == "explore" or is_starving:
            # Bias random toward movement and scanning (not interact/noop)
            return int(rng.choice(_MOVE_ACTIONS))
        return int(rng.integers(0, n_actions))

    # Rule 2: Energy critically low -> seek resources (always priority).
    # Cascade: visible resources > dirt cleaning > memory > heatmap > growth > entropy.
    if state.energy < LOW_ENERGY_THRESHOLD:
        if perception.resources_nearby:
            return _move_toward(perception.resource_direction, rng, heading)
        elif perception.dirt_nearby:
            # No food but pollution visible — clean to restart apple growth.
            if perception.dirt_density > DIRT_CLOSE_DENSITY:
                return interact_action
            return _move_toward(perception.dirt_direction, rng, heading)
        elif _has_memory(state):
            return _move_toward(state.resource_memory, rng, heading)
        elif _has_heatmap(state):
            return _move_toward(_grid_gradient(state.resource_heatmap), rng, heading)
        elif _has_growth(perception):
            return _move_toward(perception.growth_gradient, rng, heading)
        else:
            return _move_toward(perception.gradient, rng, heading)

    # Rule 2.5: Dirt cleaning — approach pollution and INTERACT to clean.
    # In Clean Up, apple growth drops to ZERO when river pollution exceeds 40%.
    # If we see dirt but no resources, apples have likely stopped growing.
    # Echo feedback: when INTERACT on dirt causes change, cooperation rises
    # (already in state machine). High-cooperation agents keep cleaning even
    # when some food appears — committed cleaners sustain apple regrowth.
    should_clean = perception.dirt_nearby and (
        not perception.resources_nearby
        or state.cooperation_tendency > ECHO_CLEANING_THRESHOLD
    )
    if should_clean:
        if perception.dirt_density > DIRT_CLOSE_DENSITY:
            return interact_action  # close enough — fire cleaning beam
        return _move_toward(perception.dirt_direction, rng, heading)  # approach

    # Rule 3: Exploit phase bonus — seek resources at moderate energy.
    # Heatmap added as fallback between memory and growth gradient.
    if phase == "exploit" and state.energy < EXPLOIT_ENERGY_SEEK:
        if perception.resources_nearby:
            return _move_toward(perception.resource_direction, rng, heading)
        elif _has_memory(state):
            return _move_toward(state.resource_memory, rng, heading)
        elif _has_heatmap(state):
            return _move_toward(_grid_gradient(state.resource_heatmap), rng, heading)
        elif _has_growth(perception):
            return _move_toward(perception.growth_gradient, rng, heading)

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
            return _move_away(agent_dir, rng, heading)

    # Rules 5 & 6: Environment stability determines movement strategy.
    if perception.change < STABLE_THRESHOLD:
        # Stable: follow entropy gradient. Explore phase adds scan turns.
        if phase == "explore" and rng.random() < 0.3:
            return TURN_LEFT if rng.random() < 0.5 else TURN_RIGHT
        if phase == "exploit" and perception.resources_nearby:
            return _move_toward(perception.resource_direction, rng, heading)
        # Memory fallback: if stable and no resources, follow the trail
        if phase == "exploit" and _has_memory(state):
            return _move_toward(state.resource_memory, rng, heading)
        # Crowding avoidance: steer away from agent-dense quadrants
        if perception.agent_grid.max() > CROWDING_THRESHOLD:
            return _move_away(_grid_gradient(perception.agent_grid), rng, heading)
        # Heatmap fallback: follow remembered resource locations
        if _has_heatmap(state):
            return _move_toward(_grid_gradient(state.resource_heatmap), rng, heading)
        # Growth gradient: prefer moving toward where entropy is increasing
        if _has_growth(perception):
            return _move_toward(perception.growth_gradient, rng, heading)
        return _move_toward(perception.gradient, rng, heading)
    else:
        # Chaotic: exploit current role strategy.
        return _exploit_role(state, perception, n_actions, rng)
