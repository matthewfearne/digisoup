"""Thermodynamic action selection — v8 (9-Layer Entropy Stack).

The agent knows NOTHING about agents, resources, cooperation, or reward.
Pure information thermodynamics drives all behaviour.

Entropy stack (9 layers):
  Layer 1-6: Observation entropy, gradient, change, dS/dt, temperature, d2S/dt2
  Layer 7: Entropy growth gradient (where is entropy GROWING?)
  Layer 8: Thermal gradient (decaying heat trails — where WAS entropy recently?)
  Layer 9: KL divergence (what looks ANOMALOUS?)

Thermodynamic mechanisms:
  - Sigmoid entrainment, gradient momentum, autocatalytic feedback
  - Thermal fallback, growth bias, KL anomaly boost

Boltzmann action selection: P(action) ~ exp(benefit / T)
3 behaviours: INTERACT, MOVE, SCAN

Melting Pot action space (substrate-dependent)
-----------------------------------------------
0 : no-op    1 : forward    2 : backward    3 : left
4 : right    5 : turn left  6 : turn right  7 : interact (or zap)
8 : fire_clean (Clean Up only — 9-action substrate)

INTERACT always maps to n_actions - 1 (the last action = primary interaction).
"""
from __future__ import annotations

import numpy as np

from .entropy import (
    MAX_ENTROPY,
    change_entropy,
    entropy_growth_gradient,
    entropy_production_rate,
    entropy_rate,
    fine_entropy_gradient,
    fine_entropy_grid,
    information_temperature,
    kl_divergence_grid,
    observation_entropy,
    peak_local_entropy,
    spatial_change_gradient,
    thermal_gradient,
)


# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------

NOOP = 0
FORWARD = 1
BACKWARD = 2
LEFT = 3
RIGHT = 4
TURN_LEFT = 5
TURN_RIGHT = 6
INTERACT = 7

N_ACTIONS = 8


# ---------------------------------------------------------------------------
# Sigmoid for phase-transition entrainment
# ---------------------------------------------------------------------------

def _sigmoid(x: float, centre: float = 0.0, steepness: float = 10.0) -> float:
    """Sigmoid function for phase-transition behaviour."""
    z = steepness * (x - centre)
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + np.exp(-z))


def select_action(
    obs: np.ndarray,
    prev_obs: np.ndarray | None = None,
    entropy_history: list[float] | None = None,
    gradient_history: list[np.ndarray] | None = None,
    prev_grid: np.ndarray | None = None,
    thermal_map: np.ndarray | None = None,
    echo_resonance: float = 0.0,
    echo_direction: np.ndarray | None = None,
    echo_trend: float = 0.0,
    n_actions: int = N_ACTIONS,
    personality: dict | None = None,
    step_count: int = 0,
    rng: np.random.Generator | None = None,
) -> int:
    """Choose an action via full 9-layer thermodynamic physics.

    INTERACT maps to n_actions - 1 (substrate's primary interaction).
    personality scales the three Boltzmann energies — same physics,
    different thermodynamic constants per agent.
    """
    rng = rng or np.random.default_rng()
    entropy_history = entropy_history or []
    gradient_history = gradient_history or []

    # === LAYER 1-6: Core entropy signals ===

    # Layer 1: observation entropy + peak local
    ent = observation_entropy(obs)
    peak_ent = peak_local_entropy(obs, grid_size=4)

    # Layer 2: fine-grained spatial gradient
    grad = fine_entropy_gradient(obs, grid_size=4)
    grad_strength = float(np.linalg.norm(grad))

    # Layer 3: spatial change gradient
    ch_grad = spatial_change_gradient(obs, prev_obs, grid_size=4)
    ch_strength = float(np.linalg.norm(ch_grad))
    chent = change_entropy(obs, prev_obs)

    # Layer 4: entropy rate (dS/dt)
    dS_dt = entropy_rate(entropy_history)

    # Layer 5: information temperature
    T = information_temperature(entropy_history)

    # Layer 6: entropy production rate
    d2S_dt2 = entropy_production_rate(entropy_history)

    # === COLD ENVIRONMENT SENSITIVITY (Stefan-Boltzmann) ===
    # Small entropy differences matter MORE against a cold background.
    # At ent>=2.5 (Commons Harvest): boost=0.0 — completely disabled.
    # At ent~0.8 (PD arena): boost=0.68 — gradients amplified 68%.
    contrast_boost = max(0.0, 1.0 - ent / (0.5 * MAX_ENTROPY))
    effective_grad_strength = grad_strength * (1.0 + contrast_boost)
    effective_ch_strength = ch_strength * (1.0 + contrast_boost)

    # === LAYER 7: Entropy growth gradient (where is entropy GROWING?) ===
    current_grid = fine_entropy_grid(obs, grid_size=4)
    growth_grad = entropy_growth_gradient(current_grid, prev_grid)
    growth_strength = float(np.linalg.norm(growth_grad))

    # === LAYER 8: Thermal gradient (heat-seeking in dark environments) ===
    th_grad = thermal_gradient(thermal_map) if thermal_map is not None else np.zeros(2)
    th_strength = float(np.linalg.norm(th_grad))

    # === LAYER 8b: Thermal saturation (anti-pheromone / toxin avoidance) ===
    # Thermally saturated regions (high mean, low variance) = stale zone.
    # When saturated, FLIP thermal gradient to REPULSIVE (anti-pheromone).
    th_saturation = 0.0
    if thermal_map is not None:
        th_mean = float(thermal_map.mean())
        th_std = float(thermal_map.std())
        if th_mean > 1.0:
            th_saturation = th_mean / (th_std + 0.1)
            if th_saturation > 5.0:
                th_grad = -th_grad
                th_strength = float(np.linalg.norm(th_grad))

    # === LAYER 9: KL divergence (anomaly detection) ===
    kl_grid = kl_divergence_grid(obs, grid_size=4)
    kl_max = float(kl_grid.max())

    # === ENTRAINMENT (sigmoid phase transition) ===
    entrainment = _sigmoid(dS_dt, centre=0.0, steepness=15.0)

    # === GRADIENT MOMENTUM ===
    momentum_boost = 1.0
    if len(gradient_history) >= 3:
        recent = np.mean(gradient_history[-3:], axis=0)
        recent_norm = np.linalg.norm(recent)
        current = ch_grad if ch_strength > 0.05 else grad
        current_norm = np.linalg.norm(current)
        if recent_norm > 0.05 and current_norm > 0.05:
            cos_sim = float(np.dot(recent, current) / (recent_norm * current_norm))
            momentum_boost = 0.5 + 1.0 * max(0.0, cos_sim)

    # === AUTOCATALYTIC FEEDBACK ===
    autocatalysis = 0.0
    if chent > 0.5:
        autocatalysis = min(chent / MAX_ENTROPY, 0.3)

    # === COMPUTE ENERGIES FOR BOLTZMANN DISTRIBUTION ===

    # INTERACT energy: peak complexity + activity + autocatalysis
    interact_energy = (peak_ent + chent) / (2.0 * MAX_ENTROPY) * (1.0 + contrast_boost)
    if dS_dt > 0:
        interact_energy += min(dS_dt, 0.3)
    interact_energy += autocatalysis
    interact_energy *= momentum_boost

    # Layer 7 boost: entropy GROWING nearby -> things to interact with
    if growth_strength > 0.05:
        interact_energy += min(growth_strength * 0.25, 0.2)

    # Layer 9 boost: KL anomaly -> something statistically unusual
    effective_kl_threshold = max(0.1, 0.3 - 0.2 * contrast_boost)
    if kl_max > effective_kl_threshold:
        interact_energy += min(kl_max * 0.15, 0.2)

    # Echo boost: INTERACT caused more entropy change than ambient movement
    if echo_resonance > 1.0:
        interact_energy += min((echo_resonance - 1.0) * 0.15, 0.2)

    # MOVE energy: gradient strength (both static + change, equally weighted)
    move_energy = (effective_grad_strength + effective_ch_strength) / 2.0
    move_energy *= momentum_boost
    if d2S_dt2 > 0:
        move_energy += min(d2S_dt2 * 2.0, 0.2)

    # Layer 7 boost: growth happening somewhere -> move toward it
    if growth_strength > 0.1:
        move_energy += min(growth_strength * 0.15, 0.15)

    # Layer 8 boost: thermal trail exists -> worth following
    if th_strength > 0.05:
        move_energy += min(th_strength * 0.1, 0.1)

    # === ZONE COOLING -> MIGRATION (oscillation driver) ===
    # When dS/dt is negative, current zone is cooling. Chase the growth gradient.
    cooling_pressure = 0.0
    if dS_dt < -0.02:
        cooling_pressure = min(abs(dS_dt) * 3.0, 0.4)
        move_energy += cooling_pressure
        if growth_strength > 0.05:
            move_energy += cooling_pressure * 0.3

    # SCAN energy: baseline exploration + echo decline
    scan_energy = 0.15
    if echo_trend < -0.2:
        scan_energy += min(abs(echo_trend) * 0.2, 0.15)
    if dS_dt < -0.05:
        scan_energy += min(abs(dS_dt) * 0.3, 0.10)

    # Thermal saturation: stale zone -> MOVE out
    if th_saturation > 5.0:
        saturation_effect = min((th_saturation - 5.0) / 10.0, 0.3)
        move_energy += saturation_effect
        scan_energy += saturation_effect * 0.3

    # === BOLTZMANN DISTRIBUTION (3 behaviours) ===
    p = personality or {"interact_bias": 1.0, "move_bias": 1.0, "scan_bias": 1.0}
    energies = np.array([
        interact_energy * p.get("interact_bias", 1.0),
        move_energy * p.get("move_bias", 1.0),
        scan_energy * p.get("scan_bias", 1.0),
    ])
    boltz = np.exp(energies / T)
    probs = boltz / boltz.sum()

    roll = rng.random()
    cumulative = 0.0

    # Behaviour 1: INTERACT (maps to last action = substrate's primary interaction)
    interact_action = n_actions - 1
    cumulative += probs[0]
    if roll < cumulative:
        return interact_action

    # Langevin noise std: ramps with step_count (gradual restlessness).
    noise_std = 0.3 + 0.2 * min(step_count / 600.0, 1.0)

    # Behaviour 2: MOVE toward combined gradient
    cumulative += probs[1]
    if roll < cumulative:
        return _gradient_to_movement(
            grad, ch_grad, ch_strength,
            growth_grad, growth_strength,
            th_grad, th_strength,
            echo_direction, echo_resonance,
            noise_std, cooling_pressure, rng,
        )

    # Behaviour 3: SCAN -- turn toward gradient
    return _gradient_to_turn(
        grad, ch_grad, ch_strength,
        th_grad, th_strength,
        echo_direction, echo_resonance,
        noise_std, rng,
    )


def _gradient_to_movement(
    grad: np.ndarray,
    ch_grad: np.ndarray,
    ch_strength: float,
    growth_grad: np.ndarray,
    growth_strength: float,
    th_grad: np.ndarray,
    th_strength: float,
    echo_direction: np.ndarray | None,
    echo_resonance: float,
    noise_std: float,
    cooling_pressure: float,
    rng: np.random.Generator,
) -> int:
    """Convert thermodynamic gradients to movement.

    Uses change gradient when available (stuff is happening),
    falls back to static gradient (where is complexity?).
    Growth bias amplified when zone is cooling.
    """
    # Use change gradient if meaningful change detected, else static
    if ch_strength > 0.05:
        primary = ch_grad
    else:
        primary = grad

    primary_strength = float(np.linalg.norm(primary))

    # Thermal fallback: sigmoid transition -- kicks in when primary is weak
    thermal_reliance = _sigmoid(-primary_strength, centre=-0.08, steepness=30.0)

    # Growth bias: where entropy is INCREASING (apple regrowth, dirt accumulating).
    # Amplified when zone is cooling (dS/dt negative).
    growth_scale = 0.4 + cooling_pressure * 1.5
    growth_bias = growth_grad * growth_scale if growth_strength > 0.05 else np.zeros(2)

    # Echo chase: toward reactive zones (my interactions caused more change)
    echo_chase = np.zeros(2)
    if echo_direction is not None:
        echo_dir_strength = float(np.linalg.norm(echo_direction))
        if echo_dir_strength > 0.02 and echo_resonance > 1.0:
            echo_chase = echo_direction * min(echo_resonance * 0.3, 0.5)

    # Combine: primary + growth + echo + thermal
    combined = (
        primary * (1.0 - thermal_reliance)
        + th_grad * thermal_reliance
        + growth_bias
        + echo_chase
    )

    # Langevin noise: Brownian thermal kicks for dispersal.
    combined = combined + rng.normal(0, noise_std, size=2)

    norm = np.linalg.norm(combined)
    if norm < 0.05:
        return FORWARD

    dy, dx = float(combined[0]), float(combined[1])
    if abs(dy) >= abs(dx):
        if dy < 0:
            return FORWARD
        else:
            if rng.random() < 0.5:
                return TURN_LEFT if rng.random() < 0.5 else TURN_RIGHT
            else:
                return BACKWARD
    else:
        return RIGHT if dx > 0 else LEFT


def _gradient_to_turn(
    grad: np.ndarray,
    ch_grad: np.ndarray,
    ch_strength: float,
    th_grad: np.ndarray,
    th_strength: float,
    echo_direction: np.ndarray | None,
    echo_resonance: float,
    noise_std: float,
    rng: np.random.Generator,
) -> int:
    """Convert thermodynamic gradients to turn direction.

    Turn toward change (stuff happening) or static entropy (complexity).
    Echo chase: turn toward where interactions were productive.
    """
    if ch_strength > 0.05:
        primary = ch_grad
    else:
        primary = grad

    primary_strength = float(np.linalg.norm(primary))
    thermal_reliance = _sigmoid(-primary_strength, centre=-0.08, steepness=30.0)

    # Echo chase: turn toward reactive zones
    echo_chase = np.zeros(2)
    if echo_direction is not None:
        echo_dir_strength = float(np.linalg.norm(echo_direction))
        if echo_dir_strength > 0.02 and echo_resonance > 1.0:
            echo_chase = echo_direction * min(echo_resonance * 0.3, 0.5)

    combined = (
        primary * (1.0 - thermal_reliance)
        + th_grad * thermal_reliance
        + echo_chase
    )

    combined = combined + rng.normal(0, noise_std * 0.7, size=2)

    norm = np.linalg.norm(combined)
    if norm < 0.05:
        return TURN_LEFT if rng.random() < 0.5 else TURN_RIGHT

    dy, dx = float(combined[0]), float(combined[1])
    if abs(dx) > abs(dy):
        return TURN_RIGHT if dx > 0 else TURN_LEFT
    if dy < 0:
        return FORWARD
    return TURN_LEFT if rng.random() < 0.5 else TURN_RIGHT
