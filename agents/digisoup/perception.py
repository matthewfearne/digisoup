"""Thermodynamic perception from RGB observations.

v10 "Sharper Eyes" upgrades:
- Warm mask: detects red/orange apples (CH/CU) previously misclassified as agents
- Dirt mask: detects CU pollution for cleanup scenarios
- Resource mask: green + warm = all apple types
- Agent mask: now excludes resources (fixes apple-fleeing bug)
- Agent density grid: 4x4 spatial map of agent locations
- Dirt detection: direction and density of CU pollution

v8 base layers:
- Fine-grained 4x4 entropy gradient (replaces 2x2 quadrant gradient)
- Entropy growth gradient: direction toward where entropy is INCREASING (dS/dt)
- KL divergence anomaly detection: finds statistically improbable patches

NOT a convolutional neural network. Histogram-based features only.
Every component explainable in one paragraph.
"""
from __future__ import annotations

import numpy as np
from typing import NamedTuple

_BINS = 32
_LOG2_EPS = 1e-12
MAX_ENTROPY = np.log2(_BINS)  # ~5.0 bits


_GRID_SIZE = 4  # 4x4 grid for fine-grained spatial entropy


class Perception(NamedTuple):
    """What the agent perceives from its RGB observation."""
    entropy: float                 # Shannon entropy of full observation
    gradient: np.ndarray           # (dy, dx) toward higher-entropy region (4x4 grid)
    entropy_grid: np.ndarray       # 4x4 grid of per-patch entropy values
    growth_gradient: np.ndarray    # (dy, dx) toward where entropy is INCREASING
    anomaly_direction: np.ndarray  # (dy, dx) toward most anomalous patch (KL)
    anomaly_strength: float        # max KL divergence value (0 = uniform)
    agents_nearby: bool            # agent-coloured pixels detected
    agent_direction: np.ndarray    # (dy, dx) toward agents
    agent_density: float           # fraction of agent pixels
    agent_grid: np.ndarray         # (4,4) agent density per spatial cell
    resources_nearby: bool         # resource-coloured pixels detected
    resource_direction: np.ndarray # (dy, dx) toward resources
    resource_density: float        # fraction of resource pixels
    dirt_nearby: bool              # CU pollution detected
    dirt_direction: np.ndarray     # (dy, dx) toward pollution
    dirt_density: float            # fraction of dirt pixels
    growth_rate: float             # mean entropy change rate (+ = growing, - = depleting)
    change: float                  # frame-to-frame change entropy
    change_direction: np.ndarray   # (dy, dx) toward area of most change


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def _channel_entropy(pixels: np.ndarray) -> float:
    """Shannon entropy (bits) of a flat array of uint8 pixel values."""
    hist, _ = np.histogram(pixels, bins=_BINS, range=(0, 256))
    total = hist.sum()
    if total == 0:
        return 0.0
    pmf = hist / total
    pmf = pmf[pmf > 0]
    return max(0.0, float(-np.sum(pmf * np.log2(pmf + _LOG2_EPS))))


def _patch_entropy(patch: np.ndarray) -> float:
    """Shannon entropy of an RGB image patch, mean across channels."""
    if patch.ndim != 3 or patch.shape[2] < 3 or patch.size == 0:
        return 0.0
    return (
        _channel_entropy(patch[:, :, 0].ravel()) +
        _channel_entropy(patch[:, :, 1].ravel()) +
        _channel_entropy(patch[:, :, 2].ravel())
    ) / 3.0


# ---------------------------------------------------------------------------
# Fine-grained entropy grid (4x4, replaces 2x2 quadrant gradient)
# ---------------------------------------------------------------------------

def _fine_entropy_grid(obs: np.ndarray) -> np.ndarray:
    """Compute entropy on a 4x4 grid of patches.

    Returns shape (4, 4) array of entropy values. Each cell is the Shannon
    entropy of that patch. 4x finer than quadrants — better directional info.
    """
    if obs.ndim != 3:
        return np.zeros((_GRID_SIZE, _GRID_SIZE))
    h, w = obs.shape[0], obs.shape[1]
    ph = max(h // _GRID_SIZE, 1)
    pw = max(w // _GRID_SIZE, 1)
    grid = np.zeros((_GRID_SIZE, _GRID_SIZE))
    for gy in range(_GRID_SIZE):
        for gx in range(_GRID_SIZE):
            y0, x0 = gy * ph, gx * pw
            y1, x1 = min(y0 + ph, h), min(x0 + pw, w)
            grid[gy, gx] = _patch_entropy(obs[y0:y1, x0:x1])
    return grid


def _grid_gradient(grid: np.ndarray) -> np.ndarray:
    """Direction toward highest values in a spatial grid.

    Weighted centre of mass relative to grid centre. Returns (dy, dx)
    unit vector. Works for entropy grids, KL grids, growth grids, etc.
    """
    grid_size = grid.shape[0]
    if grid.max() - grid.min() < 1e-6:
        return np.zeros(2)

    centre = (grid_size - 1) / 2.0
    weights = grid - grid.mean()

    total_weight = np.abs(weights).sum()
    if total_weight < 1e-8:
        return np.zeros(2)

    dy, dx = 0.0, 0.0
    for gy in range(grid_size):
        for gx in range(grid_size):
            dy += (gy - centre) * weights[gy, gx]
            dx += (gx - centre) * weights[gy, gx]

    grad = np.array([dy, dx])
    norm = np.linalg.norm(grad)
    if norm < 1e-8:
        return np.zeros(2)
    if norm > 1.0:
        return grad / norm
    return grad


# ---------------------------------------------------------------------------
# Entropy growth gradient (where is entropy INCREASING?)
# ---------------------------------------------------------------------------

def _entropy_growth_gradient(
    current_grid: np.ndarray,
    prev_grid: np.ndarray | None,
) -> np.ndarray:
    """Direction toward where entropy is growing (positive dS/dt).

    Points toward apple regrowth, agent arrival, new activity.
    The thermodynamic arrow of time: flow toward entropy production.
    Returns (dy, dx) vector.
    """
    if prev_grid is None or current_grid.shape != prev_grid.shape:
        return np.zeros(2)

    rate = current_grid - prev_grid
    # Only consider POSITIVE rate (growth). Zero out decay.
    growth = np.maximum(rate, 0.0)
    if growth.max() < 1e-6:
        return np.zeros(2)

    grid_size = growth.shape[0]
    centre = (grid_size - 1) / 2.0

    total_weight = growth.sum()
    if total_weight < 1e-8:
        return np.zeros(2)

    dy, dx = 0.0, 0.0
    for gy in range(grid_size):
        for gx in range(grid_size):
            dy += (gy - centre) * growth[gy, gx]
            dx += (gx - centre) * growth[gy, gx]

    grad = np.array([dy, dx])
    norm = np.linalg.norm(grad)
    if norm < 1e-8:
        return np.zeros(2)
    if norm > 1.0:
        return grad / norm
    return grad


# ---------------------------------------------------------------------------
# KL divergence anomaly detection
# ---------------------------------------------------------------------------

def _kl_divergence_grid(obs: np.ndarray) -> np.ndarray:
    """KL divergence of each grid cell's pixel distribution vs the global.

    High KL = this patch looks DIFFERENT from the background.
    In a dark arena, an agent's coloured sprite creates massive KL divergence.
    Pure information theory anomaly detection — not object recognition.
    """
    if obs.ndim != 3:
        return np.zeros((_GRID_SIZE, _GRID_SIZE))

    global_hist, _ = np.histogram(obs.ravel(), bins=_BINS, range=(0, 256))
    global_total = global_hist.sum()
    if global_total == 0:
        return np.zeros((_GRID_SIZE, _GRID_SIZE))
    global_pmf = global_hist / global_total
    global_pmf = np.maximum(global_pmf, _LOG2_EPS)

    h, w = obs.shape[0], obs.shape[1]
    ph, pw = max(h // _GRID_SIZE, 1), max(w // _GRID_SIZE, 1)
    kl_grid = np.zeros((_GRID_SIZE, _GRID_SIZE))

    for gy in range(_GRID_SIZE):
        for gx in range(_GRID_SIZE):
            y0, x0 = gy * ph, gx * pw
            y1, x1 = min(y0 + ph, h), min(x0 + pw, w)
            patch = obs[y0:y1, x0:x1]

            local_hist, _ = np.histogram(patch.ravel(), bins=_BINS, range=(0, 256))
            local_total = local_hist.sum()
            if local_total == 0:
                continue
            local_pmf = local_hist / local_total
            local_pmf = np.maximum(local_pmf, _LOG2_EPS)

            kl = float(np.sum(local_pmf * np.log2(local_pmf / global_pmf)))
            kl_grid[gy, gx] = max(0.0, kl)

    return kl_grid


# ---------------------------------------------------------------------------
# Pixel detection helpers
# ---------------------------------------------------------------------------

def _pixel_direction(obs: np.ndarray, mask: np.ndarray) -> tuple[bool, np.ndarray, float]:
    """Compute direction and density from a boolean pixel mask.

    Returns (detected, direction_unit_vector, density).
    """
    total_pixels = mask.size
    count = int(mask.sum())
    density = count / total_pixels if total_pixels > 0 else 0.0

    if count < 5:
        return False, np.zeros(2), 0.0

    ys, xs = np.where(mask)
    cy, cx = obs.shape[0] / 2.0, obs.shape[1] / 2.0
    dy = float(np.mean(ys) - cy)
    dx = float(np.mean(xs) - cx)

    direction = np.array([dy, dx])
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        direction = direction / norm

    return True, direction, density


def _green_mask(obs: np.ndarray) -> np.ndarray:
    """Detect green-ish pixels (resources/apples in most substrates)."""
    r = obs[:, :, 0].astype(np.float32)
    g = obs[:, :, 1].astype(np.float32)
    b = obs[:, :, 2].astype(np.float32)
    return (g > 100) & (g > r * 1.3) & (g > b * 1.3)


def _warm_mask(obs: np.ndarray) -> np.ndarray:
    """Detect warm-coloured resource pixels (red/orange apples).

    Catches CH red apples (214,88,88) and CU orange apples (212,80,57).
    Excludes all 7 Melting Pot agent palette colours by requiring
    r>160, g in [40,120], and large r-g gap.
    """
    r = obs[:, :, 0].astype(np.float32)
    g = obs[:, :, 1].astype(np.float32)
    b = obs[:, :, 2].astype(np.float32)
    return (r > 160) & (g >= 40) & (g < 120) & (b > 25) & ((r - g) > 60)


def _dirt_mask(obs: np.ndarray) -> np.ndarray:
    """Detect river/water pixels as proxy for 'cleanable area'.

    CU pollution is (2,245,80) at 20% alpha over water, making it nearly
    identical to clean water visually. We detect ALL river water instead —
    the agent should FIRE_CLEAN when near any river tile. Cleaning clean
    water is harmless; cleaning dirty water removes pollution.

    River water colors: (27-35, 125-185, 143-175) — teal/cyan with low red.
    Same as _water_mask — kept in sync for agent_mask exclusion.
    """
    r = obs[:, :, 0].astype(np.float32)
    g = obs[:, :, 1].astype(np.float32)
    b = obs[:, :, 2].astype(np.float32)
    return (r < 60) & (g > 100) & (b > 100)


def _water_mask(obs: np.ndarray) -> np.ndarray:
    """Detect river/water pixels (teal/cyan with very low red).

    CU river water is ~(27,150,144) — low r, moderate-high g and b.
    Previously misdetected as agents due to high saturation (123).
    """
    r = obs[:, :, 0].astype(np.float32)
    g = obs[:, :, 1].astype(np.float32)
    b = obs[:, :, 2].astype(np.float32)
    return (r < 60) & (g > 100) & (b > 100)


def _resource_mask(obs: np.ndarray) -> np.ndarray:
    """Detect all resource pixels (green apples + red/orange apples)."""
    return _green_mask(obs) | _warm_mask(obs)


def _agent_mask(obs: np.ndarray) -> np.ndarray:
    """Detect agent-coloured pixels (saturated, non-resource/water/dirt, not too dark).

    v15 fix: excludes river water and dirt that were previously misdetected
    as agents, causing phantom agent sightings near the CU river.
    """
    r = obs[:, :, 0].astype(np.float32)
    g = obs[:, :, 1].astype(np.float32)
    b = obs[:, :, 2].astype(np.float32)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    saturation = max_c - min_c

    exclude = _resource_mask(obs) | _water_mask(obs) | _dirt_mask(obs)
    return (saturation > 50) & (~exclude) & (max_c > 80)


def _agent_density_grid(obs: np.ndarray, agent_mask_result: np.ndarray) -> np.ndarray:
    """Compute 4x4 grid of agent density per spatial cell."""
    h, w = obs.shape[:2]
    gh, gw = max(h // _GRID_SIZE, 1), max(w // _GRID_SIZE, 1)
    grid = np.zeros((_GRID_SIZE, _GRID_SIZE))
    for gy in range(_GRID_SIZE):
        for gx in range(_GRID_SIZE):
            y0, x0 = gy * gh, gx * gw
            y1, x1 = min(y0 + gh, h), min(x0 + gw, w)
            patch = agent_mask_result[y0:y1, x0:x1]
            total = patch.size
            if total > 0:
                grid[gy, gx] = float(patch.sum()) / total
    return grid


# ---------------------------------------------------------------------------
# Main perception function
# ---------------------------------------------------------------------------

def perceive(
    obs: np.ndarray,
    prev_obs: np.ndarray | None = None,
    prev_entropy_grid: np.ndarray | None = None,
) -> Perception:
    """Process an 88x88x3 RGB observation into perception signals.

    All computation is histogram-based pixel statistics. No learned features.
    prev_entropy_grid: previous frame's 4x4 entropy grid (for growth gradient).
    """
    obs = np.asarray(obs, dtype=np.uint8)
    if obs.ndim == 2:
        obs = np.stack([obs] * 3, axis=-1)

    # Shannon entropy of full observation
    entropy = _patch_entropy(obs)

    # Fine-grained 4x4 entropy grid + gradient (replaces 2x2 quadrants)
    entropy_grid = _fine_entropy_grid(obs)
    gradient = _grid_gradient(entropy_grid)

    # Entropy growth gradient: where is entropy INCREASING? (dS/dt)
    growth_gradient = _entropy_growth_gradient(entropy_grid, prev_entropy_grid)

    # Growth rate: scalar mean entropy change (positive = richer, negative = depleting)
    if prev_entropy_grid is not None and entropy_grid.shape == prev_entropy_grid.shape:
        growth_rate = float(np.mean(entropy_grid - prev_entropy_grid))
    else:
        growth_rate = 0.0

    # KL divergence anomaly detection
    kl_grid = _kl_divergence_grid(obs)
    anomaly_direction = _grid_gradient(kl_grid)
    anomaly_strength = float(kl_grid.max())

    # Agent detection by colour signature (v10: excludes warm resources)
    agent_mask_result = _agent_mask(obs)
    agents_nearby, agent_direction, agent_density = _pixel_direction(
        obs, agent_mask_result
    )
    agent_grid = _agent_density_grid(obs, agent_mask_result)

    # Resource detection (v10: green + warm = all apple colours)
    resources_nearby, resource_direction, resource_density = _pixel_direction(
        obs, _resource_mask(obs)
    )

    # Dirt/pollution detection (CU cleanup substrate)
    dirt_nearby, dirt_direction, dirt_density = _pixel_direction(
        obs, _dirt_mask(obs)
    )

    # Change detection via observation differencing
    change = 0.0
    change_direction = np.zeros(2)
    if prev_obs is not None and obs.shape == prev_obs.shape:
        diff = np.abs(
            obs.astype(np.int16) - prev_obs.astype(np.int16)
        ).astype(np.uint8)
        change = _patch_entropy(diff)
        # Direction toward pixels that changed the most
        change_mask = np.max(diff, axis=2) > 20
        if change_mask.any():
            _, change_direction, _ = _pixel_direction(obs, change_mask)

    return Perception(
        entropy=entropy,
        gradient=gradient,
        entropy_grid=entropy_grid,
        growth_gradient=growth_gradient,
        anomaly_direction=anomaly_direction,
        anomaly_strength=anomaly_strength,
        agents_nearby=agents_nearby,
        agent_direction=agent_direction,
        agent_density=agent_density,
        agent_grid=agent_grid,
        resources_nearby=resources_nearby,
        resource_direction=resource_direction,
        resource_density=resource_density,
        dirt_nearby=dirt_nearby,
        dirt_direction=dirt_direction,
        dirt_density=dirt_density,
        growth_rate=growth_rate,
        change=change,
        change_direction=change_direction,
    )
