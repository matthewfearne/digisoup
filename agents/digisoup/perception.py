"""Simple perception from RGB observations.

88x88x3 RGB observations processed with SIMPLE pixel statistics:
- Shannon entropy of pixel distribution
- Gradient across observation quadrants
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


class Perception(NamedTuple):
    """What the agent perceives from its RGB observation."""
    entropy: float                 # Shannon entropy of full observation
    gradient: np.ndarray           # (dy, dx) toward higher-entropy quadrant
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
# Gradient (quadrant-based)
# ---------------------------------------------------------------------------

def _quadrant_gradient(obs: np.ndarray) -> np.ndarray:
    """Entropy gradient across 4 quadrants. Returns (dy, dx) unit vector.

    Divides the observation into four quadrants, computes entropy of each,
    and returns a direction vector pointing toward the higher-entropy side.
    """
    h, w = obs.shape[0], obs.shape[1]
    mh, mw = h // 2, w // 2

    tl = _patch_entropy(obs[:mh, :mw])
    tr = _patch_entropy(obs[:mh, mw:])
    bl = _patch_entropy(obs[mh:, :mw])
    br = _patch_entropy(obs[mh:, mw:])

    dy = (bl + br) - (tl + tr)  # positive = downward
    dx = (tr + br) - (tl + bl)  # positive = rightward

    grad = np.array([dy, dx])
    norm = np.linalg.norm(grad)
    if norm < 1e-6:
        return np.zeros(2)
    return grad / norm


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

def perceive(obs: np.ndarray, prev_obs: np.ndarray | None = None) -> Perception:
    """Process an 88x88x3 RGB observation into perception signals.

    All computation is histogram-based pixel statistics. No learned features.
    """
    obs = np.asarray(obs, dtype=np.uint8)
    if obs.ndim == 2:
        obs = np.stack([obs] * 3, axis=-1)

    # Shannon entropy of full observation
    entropy = _patch_entropy(obs)

    # Direction toward higher entropy quadrant
    gradient = _quadrant_gradient(obs)

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
        agents_nearby=agents_nearby,
        agent_direction=agent_direction,
        agent_density=agent_density,
        resources_nearby=resources_nearby,
        resource_direction=resource_direction,
        resource_density=resource_density,
        change=change,
        change_direction=change_direction,
    )
