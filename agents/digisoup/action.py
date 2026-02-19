"""Entropy-gradient-based action selection for DigiSoup agents.

v15 "River Eyes":
- Fixed colour detection: water, grass, sand masks prevent phantom agents
- FIRE_CLEAN (action 8) for Clean Up 9-action space
- Sand avoidance + grass attraction for orchard navigation
- Hive mind (v14), cleaning rule (v11), heatmap/heading (v10), 4x4 grid (v8)

Priority rules:
1. Random exploration — higher in explore phase, lower in exploit
2. Energy critically low -> seek food; if depleted (dS/dt<=0) + no food -> go clean river
2.5. River cleaning — AT river (>15% water) always clean; approaching only if no food
2.7. Proactive cleaning — environment depleting + no food visible -> approach river
3. Exploit phase: seek resources at moderate energy (memory/heatmap/growth/hive)
4. Agents nearby -> symbiosis: near river=join cleaning, crowded=complement, else=cooperate/flee
5. Stable environment -> sand flee / grass attract / avoid crowds / heatmap / growth / hive
6. Chaotic environment -> exploit current role

NO reward optimization. NO training. Every rule explainable in one paragraph.

Melting Pot action space:
  0: no-op  1: forward  2: backward  3: left
  4: right  5: turn left  6: turn right  7: fireZap  8: fireClean (CU only)
"""
from __future__ import annotations

import numpy as np

from .perception import Perception, MAX_ENTROPY, _grid_gradient
from .state import (
    DigiSoupState, LOW_ENERGY_THRESHOLD, get_role, get_phase,
    HEATMAP_THRESHOLD, HEADING_BLEND, HEADING_MIN_NORM,
    world_to_ego,
)


# ---------------------------------------------------------------------------
# Action constants (Melting Pot action space)
# ---------------------------------------------------------------------------

NOOP = 0
FORWARD = 1
BACKWARD = 2
LEFT = 3
RIGHT = 4
TURN_LEFT = 5
TURN_RIGHT = 6
INTERACT = 7      # fireZap (standard 8-action space)
FIRE_CLEAN = 8    # fireClean (Clean Up 9-action space)

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

# Sand avoidance — dead zone with nothing useful
SAND_FLEE_DENSITY = 0.05       # sand density above this = flee toward productive area

# Grass attraction — orchard floor where apples grow
GRASS_ATTRACT_DENSITY = 0.02   # grass density above this = move toward orchard

# River cleaning (Clean Up substrate: pollution blocks apple growth)
DIRT_APPROACH_DENSITY = 0.03  # need 3% of view to be water before approaching river
DIRT_CLOSE_DENSITY = 0.08    # need 8% of view to be water before firing FIRE_CLEAN
                             # (sand guard prevents false firing — no need for high threshold)

# Depleted environment — apples not regrowing, river needs cleaning
DEPLETED_GROWTH_THRESHOLD = 0.0  # growth_rate at or below this = environment depleting


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
        return INTERACT  # always 7 (fireZap — social interaction)
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
    hive_direction: np.ndarray | None = None,
) -> int:
    """Choose action using phase-modulated entropy-gradient priority rules.

    The agent alternates between explore and exploit phases on a fixed
    clock. Spatial memory (slime mold path reinforcement) biases movement
    toward remembered resource locations when nothing is currently visible.
    Resource heatmap provides a broader spatial memory fallback.
    Heading persistence smooths movement trajectories.
    Hive direction: shared discovery from other agents (world coords).
    """
    rng = rng or np.random.default_rng()
    interact_action = INTERACT  # always 7 (fireZap — social interaction)
    # Clean Up has 9 actions: action 8 = FIRE_CLEAN. Other substrates use 8.
    clean_action = FIRE_CLEAN if n_actions > 8 else interact_action
    phase = get_phase(state)
    heading = state.heading

    # Convert hive direction from world to egocentric
    hive_ego = None
    if hive_direction is not None:
        ego = world_to_ego(hive_direction, state.orientation)
        if np.linalg.norm(ego) > 0.05:
            hive_ego = ego

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

    # Sand position check — used by multiple rules to prevent false river actions.
    # FOV is 11 tiles on a 30-tile map; agents in sand can see river at the edge.
    not_in_sand = not (perception.sand_nearby and perception.sand_density > SAND_FLEE_DENSITY)

    # Rule 2: Energy critically low -> seek resources (always priority).
    # Cascade: visible food > memory > heatmap > depleted→navigate to river > fallbacks.
    # If no food found AND environment depleting, HEAD TOWARD the river — apples won't
    # regrow until pollution is removed. Thermodynamic insight: dS/dt <= 0 = dying.
    # Navigation toward river allowed from anywhere; only FIRING blocked in sand.
    if state.energy < LOW_ENERGY_THRESHOLD:
        if perception.resources_nearby:
            return _move_toward(perception.resource_direction, rng, heading)
        elif _has_memory(state):
            return _move_toward(state.resource_memory, rng, heading)
        elif _has_heatmap(state):
            return _move_toward(_grid_gradient(state.resource_heatmap), rng, heading)
        elif (perception.growth_rate <= DEPLETED_GROWTH_THRESHOLD
              and perception.dirt_nearby):
            # No food, environment depleting → river polluted → go clean
            if perception.dirt_density > DIRT_CLOSE_DENSITY and not_in_sand:
                return clean_action  # at the river — fire beam
            return _move_toward(perception.dirt_direction, rng, heading)  # walk toward river
        elif perception.sand_nearby and perception.sand_density > SAND_FLEE_DENSITY:
            return _move_away(perception.sand_direction, rng, heading)
        elif perception.grass_nearby and perception.grass_density > GRASS_ATTRACT_DENSITY:
            return _move_toward(perception.grass_direction, rng, heading)
        elif _has_growth(perception):
            return _move_toward(perception.growth_gradient, rng, heading)
        elif hive_ego is not None:
            return _move_toward(hive_ego, rng, heading)
        else:
            return _move_toward(perception.gradient, rng, heading)

    # Rule 2.5: River cleaning — critical for Clean Up apple growth.
    # AT the river (>15% water, not in sand): always fire cleaning beam.
    # Approaching (>5% water, no food visible): walk toward river from anywhere.
    # Apple growth drops to ZERO when river pollution exceeds 40%.
    if (perception.dirt_nearby and perception.dirt_density > DIRT_CLOSE_DENSITY
            and not_in_sand):
        return clean_action  # at the river — fire cleaning beam
    if (perception.dirt_nearby and not perception.resources_nearby
            and perception.dirt_density > DIRT_APPROACH_DENSITY):
        return _move_toward(perception.dirt_direction, rng, heading)  # approach river

    # Rule 2.7: Proactive cleaning — environment depleting, invest in public good.
    # If NOT hungry but no food visible and growth_rate <= 0, river is likely polluted.
    # Navigate toward river from anywhere; only fire beam when actually there.
    if (perception.growth_rate <= DEPLETED_GROWTH_THRESHOLD
            and not perception.resources_nearby
            and perception.dirt_nearby):
        if perception.dirt_density > DIRT_CLOSE_DENSITY and not_in_sand:
            return clean_action
        return _move_toward(perception.dirt_direction, rng, heading)

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
        elif hive_ego is not None:
            return _move_toward(hive_ego, rng, heading)

    # Rule 4: Agents nearby — context-aware symbiosis.
    # Instead of always zapping or fleeing, respond based on environment:
    # - Near river + agents → join cleaning crew (CU symbiosis)
    # - Crowded orchard + river visible → complement: go clean instead of competing
    # - No river context → phase-dependent cooperate/flee (PD/CH behavior)
    # Note: fireZap PUNISHES other agents (freezes them). In CU this hurts
    # everyone — frozen bots can't clean. Only zap when no river context.
    agents_detected = perception.agents_nearby
    agent_dir = perception.agent_direction
    if not agents_detected and perception.anomaly_strength > ANOMALY_AGENT_THRESHOLD:
        agents_detected = True
        agent_dir = perception.anomaly_direction

    if agents_detected:
        if perception.dirt_nearby and not_in_sand:
            # Near river with other agents → join cleaning effort
            if perception.dirt_density > DIRT_CLOSE_DENSITY:
                return clean_action  # clean alongside others
            return _move_toward(perception.dirt_direction, rng, heading)
        elif (perception.dirt_nearby
              and perception.agent_grid.max() > CROWDING_THRESHOLD):
            # Crowded area, river visible at distance → go clean (complement)
            return _move_toward(perception.dirt_direction, rng, heading)
        else:
            # No river context → original cooperate/flee for PD/CH
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
        # Sand avoidance: flee dead zones toward productive areas
        if perception.sand_nearby and perception.sand_density > SAND_FLEE_DENSITY:
            return _move_away(perception.sand_direction, rng, heading)
        # Grass attraction: move toward orchard (where apples grow)
        if perception.grass_nearby and perception.grass_density > GRASS_ATTRACT_DENSITY:
            return _move_toward(perception.grass_direction, rng, heading)
        # Crowding avoidance: steer away from agent-dense quadrants
        if perception.agent_grid.max() > CROWDING_THRESHOLD:
            return _move_away(_grid_gradient(perception.agent_grid), rng, heading)
        # Heatmap fallback: follow remembered resource locations
        if _has_heatmap(state):
            return _move_toward(_grid_gradient(state.resource_heatmap), rng, heading)
        # Growth gradient: prefer moving toward where entropy is increasing
        if _has_growth(perception):
            return _move_toward(perception.growth_gradient, rng, heading)
        # Hive memory: follow direction toward another agent's discovery
        if hive_ego is not None:
            return _move_toward(hive_ego, rng, heading)
        return _move_toward(perception.gradient, rng, heading)
    else:
        # Chaotic: exploit current role strategy.
        return _exploit_role(state, perception, n_actions, rng)
