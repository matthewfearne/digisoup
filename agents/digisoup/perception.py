"""Thermodynamic perception from RGB observations.

v8 upgrades perception with three information-theoretic layers:
- Fine-grained 4x4 entropy gradient (replaces 2x2 quadrant gradient)
- Entropy growth gradient: direction toward where entropy is INCREASING (dS/dt)
- KL divergence anomaly detection: finds statistically improbable patches

88x88x3 RGB observations processed with pixel statistics:
- Shannon entropy of pixel distribution (4x4 grid + full observation)
- Fine-grained entropy gradient (4x4 grid, 16 patches)
- Entropy growth gradient (where is entropy increasing? -> apple regrowth)
- KL divergence anomaly gradient (where are unusual patches? -> agents in dark)
- Agent detection by colour signature
- Resource detection by pixel patterns
- Change detection via observation differencing

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
    resources_nearby: bool         # resource-coloured pixels detected
    resource_direction: np.ndarray # (dy, dx) toward resources
    resource_density: float        # fraction of resource pixels
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


def _agent_mask(obs: np.ndarray) -> np.ndarray:
    """Detect agent-coloured pixels (saturated, non-green, not too dark)."""
    r = obs[:, :, 0].astype(np.float32)
    g = obs[:, :, 1].astype(np.float32)
    b = obs[:, :, 2].astype(np.float32)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    saturation = max_c - min_c

    green = _green_mask(obs)
    return (saturation > 50) & (~green) & (max_c > 80)


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

    # KL divergence anomaly detection
    kl_grid = _kl_divergence_grid(obs)
    anomaly_direction = _grid_gradient(kl_grid)
    anomaly_strength = float(kl_grid.max())

    # Agent detection by colour signature
    agents_nearby, agent_direction, agent_density = _pixel_direction(
        obs, _agent_mask(obs)
    )

    # Resource detection by green pixel patterns
    resources_nearby, resource_direction, resource_density = _pixel_direction(
        obs, _green_mask(obs)
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
        resources_nearby=resources_nearby,
        resource_direction=resource_direction,
        resource_density=resource_density,
        change=change,
        change_direction=change_direction,
    )
