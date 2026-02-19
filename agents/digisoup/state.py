"""Internal state machine for DigiSoup agents.

Ported from DigiSoup's biological model:
- Energy: accumulated from successful interactions, depletes over time
- Cooperation tendency: [0,1] float, shifts based on interaction outcomes
- Role: emerges from action pattern history
- Interaction history: rolling window
- Entropy state: running estimate of local environmental entropy
- Spatial memory: decaying resource direction memory (slime mold path reinforcement)

NO reward optimization. State updates use only entropy signals.
"""
from __future__ import annotations

import numpy as np
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INITIAL_ENERGY = 1.0
MAX_ENERGY = 2.0
ENERGY_DEPLETION = 0.005           # per step (lasts ~200 steps at zero income)
ENERGY_INTERACTION_GAIN = 0.1      # from successful interaction

LOW_ENERGY_THRESHOLD = 0.3         # below this = "hungry", seek resources

COOP_INITIAL = 0.5                 # start neutral
COOP_SHIFT_UP = 0.05               # after successful interaction
COOP_SHIFT_DOWN = 0.02             # after failed interaction
COOP_DECAY = 0.001                 # slow drift toward 0.5

INTERACT_CHANGE_THRESHOLD = 0.3    # change entropy needed for "successful"
ENTROPY_SMOOTHING = 0.1            # EMA factor for entropy estimate
HISTORY_LENGTH = 10                # interaction history window

INTERACT_ACTION = 7                # interact is always last standard action

# Phase cycling (jellyfish-inspired oscillation)
PHASE_LENGTH = 50                  # steps per half-cycle (full cycle = 100 steps)

# Spatial memory (slime mold path reinforcement)
MEMORY_REINFORCE = 0.3             # how strongly new resource sighting updates memory
MEMORY_DECAY = 0.05                # per-step decay of resource direction memory
RECENCY_DECAY = 0.02               # per-step decay of resource recency signal

# Resource heatmap (temporal spatial memory)
HEATMAP_DECAY = 0.95               # per-step decay multiplier
HEATMAP_REINFORCE = 0.3            # reinforcement strength per sighting
HEATMAP_THRESHOLD = 0.1            # minimum heatmap value to follow

# Directional persistence (heading EMA)
HEADING_EMA = 0.8                  # weight of previous heading
HEADING_BLEND = 0.2                # heading influence on movement direction
HEADING_MIN_NORM = 0.1             # minimum heading strength to blend

# Dead reckoning exploration (v13: spatial map)
VISIT_MAP_SIZE = 32                # 32x32 coarse grid of visit counts
VISIT_MAP_ORIGIN = 16              # agent starts at grid centre
SCOUTING_DECAY = 0.99             # per-step decay of directional interest
SCOUTING_REINFORCE = 0.3          # boost when resources/dirt detected in a direction


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class DigiSoupState(NamedTuple):
    """Internal state of a DigiSoup agent. Flows between steps."""
    step_count: int
    energy: float
    cooperation_tendency: float
    entropy_estimate: float          # running EMA of observation entropy
    prev_obs: np.ndarray             # previous observation for change detection
    prev_action: int
    interaction_outcomes: tuple       # last N interaction change values
    action_counts: tuple              # 8-tuple: count per action type
    has_prev_obs: bool                # whether prev_obs is valid
    resource_memory: np.ndarray      # decaying direction toward resources (dy, dx)
    resource_recency: float          # [0,1] how recently resources were seen
    prev_entropy_grid: np.ndarray   # previous frame's 4x4 entropy grid (for growth)
    resource_heatmap: np.ndarray    # (4,4) temporal spatial memory of resources
    heading: np.ndarray             # (2,) EMA of recent movement direction
    orientation: int                # 0=N, 1=E, 2=S, 3=W (dead reckoning)
    position: np.ndarray            # (2,) estimated world (y, x)
    visit_map: np.ndarray           # (32,32) coarse grid of visit counts
    scouting_interest: np.ndarray   # (4,) interest per world direction (N,E,S,W)


def initial_state() -> DigiSoupState:
    """Create starting state for a new episode."""
    return DigiSoupState(
        step_count=0,
        energy=INITIAL_ENERGY,
        cooperation_tendency=COOP_INITIAL,
        entropy_estimate=0.0,
        prev_obs=np.zeros((88, 88, 3), dtype=np.uint8),
        prev_action=0,
        interaction_outcomes=(),
        action_counts=(0, 0, 0, 0, 0, 0, 0, 0),
        has_prev_obs=False,
        resource_memory=np.zeros(2),
        resource_recency=0.0,
        prev_entropy_grid=np.zeros((4, 4)),
        resource_heatmap=np.zeros((4, 4)),
        heading=np.zeros(2),
        orientation=0,
        position=np.zeros(2),
        visit_map=np.zeros((VISIT_MAP_SIZE, VISIT_MAP_SIZE)),
        scouting_interest=np.zeros(4),
    )


# ---------------------------------------------------------------------------
# State update
# ---------------------------------------------------------------------------

# World direction vectors: N=up(-1,0), E=right(0,+1), S=down(+1,0), W=left(0,-1)
_WORLD_DIRS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


def _ego_to_world(ego_dir: np.ndarray, orientation: int) -> np.ndarray:
    """Convert egocentric direction to world coordinates."""
    dy, dx = float(ego_dir[0]), float(ego_dir[1])
    if orientation == 0:    # facing N
        return np.array([dy, dx])
    elif orientation == 1:  # facing E: ego-fwd = world-east
        return np.array([dx, -dy])
    elif orientation == 2:  # facing S: ego-fwd = world-south
        return np.array([-dy, -dx])
    else:                   # facing W: ego-fwd = world-west
        return np.array([-dx, dy])


def _closest_cardinal(world_dir: np.ndarray) -> int:
    """Return cardinal index (0=N,1=E,2=S,3=W) closest to direction."""
    best, best_dot = 0, -2.0
    for i, (dy, dx) in _WORLD_DIRS.items():
        dot = world_dir[0] * dy + world_dir[1] * dx
        if dot > best_dot:
            best_dot = dot
            best = i
    return best


def update_state(
    prev_state: DigiSoupState,
    obs: np.ndarray,
    action: int,
    perception_entropy: float,
    perception_change: float,
    resources_nearby: bool = False,
    resource_direction: np.ndarray | None = None,
    resource_density: float = 0.0,
    entropy_grid: np.ndarray | None = None,
    dirt_nearby: bool = False,
    dirt_direction: np.ndarray | None = None,
) -> DigiSoupState:
    """Update internal state after taking an action and receiving observation.

    Energy depletes each step. Successful interactions (entropy changed after
    INTERACT) restore energy and increase cooperation tendency. Failed
    interactions decrease cooperation tendency. This creates a feedback loop:
    agents in environments where interaction causes change become cooperators.

    Resource memory (slime mold path reinforcement): when resources are seen,
    their direction is blended into a decaying memory vector. When resources
    aren't visible, the agent can follow this memory back toward productive
    areas. Denser resource sightings reinforce more strongly.
    """
    # Entropy estimate: exponential moving average
    entropy_estimate = (
        (1 - ENTROPY_SMOOTHING) * prev_state.entropy_estimate +
        ENTROPY_SMOOTHING * perception_entropy
    )

    # Energy: deplete each step
    energy = prev_state.energy - ENERGY_DEPLETION

    # Cooperation tendency starts from previous value
    cooperation_tendency = prev_state.cooperation_tendency

    # Interaction outcomes history
    interaction_outcomes = prev_state.interaction_outcomes

    # Check if previous action was INTERACT and if it had an effect
    if prev_state.prev_action == INTERACT_ACTION and prev_state.has_prev_obs:
        if perception_change > INTERACT_CHANGE_THRESHOLD:
            # Successful interaction: entropy changed after interacting
            energy += ENERGY_INTERACTION_GAIN
            cooperation_tendency += COOP_SHIFT_UP
            interaction_outcomes = interaction_outcomes + (perception_change,)
        else:
            # Failed interaction: nothing happened
            cooperation_tendency -= COOP_SHIFT_DOWN
            interaction_outcomes = interaction_outcomes + (0.0,)

        # Trim history window
        if len(interaction_outcomes) > HISTORY_LENGTH:
            interaction_outcomes = interaction_outcomes[-HISTORY_LENGTH:]

    # Drift cooperation toward 0.5 (neutral baseline)
    cooperation_tendency += COOP_DECAY * (0.5 - cooperation_tendency)
    cooperation_tendency = max(0.0, min(1.0, cooperation_tendency))

    # Clamp energy
    energy = max(0.0, min(MAX_ENERGY, energy))

    # Update action counts
    counts = list(prev_state.action_counts)
    if 0 <= action < len(counts):
        counts[action] += 1
    action_counts = tuple(counts)

    # Spatial memory: slime mold path reinforcement
    resource_memory = prev_state.resource_memory.copy()
    resource_recency = prev_state.resource_recency

    if resources_nearby and resource_direction is not None:
        # Reinforce memory toward observed resource direction.
        # Denser patches reinforce more strongly (scale by density, capped at 1).
        strength = MEMORY_REINFORCE * min(resource_density * 10.0, 1.0)
        resource_memory = (1.0 - strength) * resource_memory + strength * resource_direction
        resource_recency = 1.0
    else:
        # Decay memory toward zero — old scent fades
        resource_memory *= (1.0 - MEMORY_DECAY)
        resource_recency = max(0.0, resource_recency - RECENCY_DECAY)

    # Resource heatmap: temporal spatial memory (4x4 grid)
    resource_heatmap = prev_state.resource_heatmap.copy() * HEATMAP_DECAY
    if resources_nearby and resource_direction is not None:
        # Map direction to grid quadrant
        gy = int(np.clip(1.5 - resource_direction[0] * 1.5, 0, 3))
        gx = int(np.clip(1.5 + resource_direction[1] * 1.5, 0, 3))
        strength = min(resource_density * 10.0, 1.0)
        resource_heatmap[gy, gx] = min(
            resource_heatmap[gy, gx] + HEATMAP_REINFORCE * strength, 1.0
        )

    # Heading: EMA of movement direction
    heading = prev_state.heading.copy()
    _ACTION_DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    action_dir = _ACTION_DIRS.get(action)
    if action_dir is not None:
        d = np.array(action_dir, dtype=np.float64)
        heading = HEADING_EMA * heading + (1.0 - HEADING_EMA) * d

    # Dead reckoning: orientation + position tracking
    orientation = prev_state.orientation
    if action == 5:     # TURN_LEFT
        orientation = (orientation - 1) % 4
    elif action == 6:   # TURN_RIGHT
        orientation = (orientation + 1) % 4

    position = prev_state.position.copy()
    if action == 1:     # FORWARD
        d = _WORLD_DIRS[prev_state.orientation]
        position += np.array(d, dtype=np.float64)
    elif action == 2:   # BACKWARD
        d = _WORLD_DIRS[(prev_state.orientation + 2) % 4]
        position += np.array(d, dtype=np.float64)
    elif action == 3:   # LEFT (strafe)
        d = _WORLD_DIRS[(prev_state.orientation - 1) % 4]
        position += np.array(d, dtype=np.float64)
    elif action == 4:   # RIGHT (strafe)
        d = _WORLD_DIRS[(prev_state.orientation + 1) % 4]
        position += np.array(d, dtype=np.float64)

    # Visit map: mark current position
    visit_map = prev_state.visit_map.copy()
    gy = int(np.clip(position[0] + VISIT_MAP_ORIGIN, 0, VISIT_MAP_SIZE - 1))
    gx = int(np.clip(position[1] + VISIT_MAP_ORIGIN, 0, VISIT_MAP_SIZE - 1))
    visit_map[gy, gx] += 1

    # Scouting interest: decay all, reinforce when resources/dirt detected
    scouting_interest = prev_state.scouting_interest.copy() * SCOUTING_DECAY
    if resources_nearby and resource_direction is not None:
        world_dir = _ego_to_world(resource_direction, prev_state.orientation)
        cardinal = _closest_cardinal(world_dir)
        scouting_interest[cardinal] += SCOUTING_REINFORCE
    if dirt_nearby and dirt_direction is not None:
        world_dir = _ego_to_world(dirt_direction, prev_state.orientation)
        cardinal = _closest_cardinal(world_dir)
        scouting_interest[cardinal] += SCOUTING_REINFORCE

    return DigiSoupState(
        step_count=prev_state.step_count + 1,
        energy=energy,
        cooperation_tendency=cooperation_tendency,
        entropy_estimate=entropy_estimate,
        prev_obs=obs.copy(),
        prev_action=action,
        interaction_outcomes=interaction_outcomes,
        action_counts=action_counts,
        has_prev_obs=True,
        resource_memory=resource_memory,
        resource_recency=resource_recency,
        prev_entropy_grid=entropy_grid if entropy_grid is not None else prev_state.prev_entropy_grid,
        resource_heatmap=resource_heatmap,
        heading=heading,
        orientation=orientation,
        position=position,
        visit_map=visit_map,
        scouting_interest=scouting_interest,
    )


# ---------------------------------------------------------------------------
# Role emergence
# ---------------------------------------------------------------------------

def get_phase(state: DigiSoupState) -> str:
    """Derive behavioral phase from step count.

    Alternates between 'explore' (discover environment, scan, move) and
    'exploit' (interact, gather, use role) every PHASE_LENGTH steps.

    Jellyfish-inspired oscillation: rhythm creates structured behavior
    from a simple clock, so the agent discovers first, then acts on
    what it found, rather than trying to do everything at once.
    """
    cycle_pos = state.step_count % (2 * PHASE_LENGTH)
    return "explore" if cycle_pos < PHASE_LENGTH else "exploit"


def get_role(state: DigiSoupState) -> str:
    """Derive emergent role from action pattern history.

    The agent doesn't choose a role — it emerges from what the agent
    has been doing. Cooperators interact a lot. Explorers move a lot.
    Scanners turn a lot. Generalists do a bit of everything.
    """
    counts = state.action_counts
    total = sum(counts)
    if total < 10:
        return "generalist"

    interact_frac = counts[INTERACT_ACTION] / total
    move_frac = (counts[1] + counts[2] + counts[3] + counts[4]) / total
    turn_frac = (counts[5] + counts[6]) / total

    if interact_frac > 0.35:
        return "cooperator"
    elif move_frac > 0.55:
        return "explorer"
    elif turn_frac > 0.35:
        return "scanner"
    return "generalist"
