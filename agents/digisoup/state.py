"""Internal state machine for DigiSoup agents.

Ported from DigiSoup's biological model:
- Energy: accumulated from successful interactions, depletes over time
- Cooperation tendency: [0,1] float, shifts based on interaction outcomes
- Role: emerges from action pattern history
- Interaction history: rolling window
- Entropy state: running estimate of local environmental entropy
- Spatial memory: decaying resource direction memory (slime mold path reinforcement)
- Entropy-as-energy: environment entropy modulates energy dynamics (cockroach persistence)

v6 adds entropy-driven energy dynamics (cockroach persistence):
Energy depletion rate scales inversely with environmental entropy — rich environments
sustain the agent, barren environments drain it faster. High entropy also provides
a small passive energy trickle (entropy IS food). At critically low energy, the agent
enters cockroach survival mode — no exploration waste, pure resource-seeking.

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
ENERGY_DEPLETION_BASE = 0.005      # base depletion per step
ENERGY_DEPLETION_MAX = 0.010       # max depletion in zero-entropy environments
ENERGY_INTERACTION_GAIN = 0.1      # from successful interaction
ENERGY_ENTROPY_TRICKLE = 0.002     # max passive energy from high entropy environments

LOW_ENERGY_THRESHOLD = 0.3         # below this = "hungry", seek resources
COCKROACH_THRESHOLD = 0.15         # below this = survival mode (cockroach persistence)

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
    )


# ---------------------------------------------------------------------------
# State update
# ---------------------------------------------------------------------------

def update_state(
    prev_state: DigiSoupState,
    obs: np.ndarray,
    action: int,
    perception_entropy: float,
    perception_change: float,
    resources_nearby: bool = False,
    resource_direction: np.ndarray | None = None,
    resource_density: float = 0.0,
) -> DigiSoupState:
    """Update internal state after taking an action and receiving observation.

    Energy depletes each step, modulated by environmental entropy (cockroach
    persistence). High-entropy environments sustain the agent (slower depletion
    + passive trickle). Low-entropy environments drain faster. Successful
    interactions restore energy and increase cooperation tendency.

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

    # Energy: entropy-modulated depletion (cockroach persistence).
    # High entropy = slower depletion + small trickle. Low entropy = faster drain.
    # entropy_estimate is in [0, ~5.0] (MAX_ENTROPY = log2(32) = 5.0)
    from .perception import MAX_ENTROPY
    entropy_ratio = min(prev_state.entropy_estimate / MAX_ENTROPY, 1.0)
    # Depletion: interpolate between max (barren) and base (rich)
    depletion = ENERGY_DEPLETION_MAX - (ENERGY_DEPLETION_MAX - ENERGY_DEPLETION_BASE) * entropy_ratio
    # Trickle: small passive gain from high-entropy environments
    trickle = ENERGY_ENTROPY_TRICKLE * entropy_ratio
    energy = prev_state.energy - depletion + trickle

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
    )


# ---------------------------------------------------------------------------
# Role emergence
# ---------------------------------------------------------------------------

def is_cockroach_mode(state: DigiSoupState) -> bool:
    """Check if agent is in cockroach survival mode.

    Below COCKROACH_THRESHOLD, the agent enters extreme conservation:
    no random exploration, no cooperation, pure resource-seeking.
    Cockroach persistence: survive in hostile environments by being
    maximally efficient. The last thing to die in any ecosystem.
    """
    return state.energy < COCKROACH_THRESHOLD


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
