"""Thermodynamic entropy stack for DigiSoup agents.

All computation is pure information theory and thermodynamics.
No neural networks. No learned features. No rules. No perception.

Layer 1: Observation entropy (how complex is what I see?)
Layer 2: Fine-grained entropy gradient (where is complexity? — high resolution)
Layer 3: Spatial change entropy gradient (where is change happening?)
Layer 4: Entropy rate dS/dt (is complexity growing or shrinking?)
Layer 5: Information temperature (how volatile is the entropy signal?)
Layer 6: Entropy production rate (thermodynamic flow direction)
Layer 7: Spatial entropy rate (dS/dt per region — where is entropy GROWING?)
         Apple regrowth = local entropy increase. Pollution clearing = local
         entropy decrease. The gradient of spatial dS/dt points toward GROWTH.
         Agents are drawn to where new things are appearing. Thermodynamic
         arrow of time: systems flow toward entropy production.
Layer 8: Thermal memory (decaying heat map — where WAS entropy recently?)
         Exponentially decaying heat map of past entropy observations.
         In dark/uniform environments where current gradients are weak,
         thermal memory provides a fallback signal to follow.

The environment speaks through entropy. Thermodynamics IS the policy.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BINS = 32          # histogram bins per channel
_LOG2_EPS = 1e-12   # avoid log(0)
MAX_ENTROPY = np.log2(_BINS)  # 5.0 bits — theoretical max for 32 bins


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _channel_entropy(pixels: np.ndarray) -> float:
    """Shannon entropy (bits) of a flat array of uint8 pixel values.

    Bins the values into _BINS equal-width buckets, normalises to a pmf,
    then computes H = -sum(p * log2(p)).
    """
    hist, _ = np.histogram(pixels, bins=_BINS, range=(0, 256))
    total = hist.sum()
    if total == 0:
        return 0.0
    pmf = hist / total
    pmf = pmf[pmf > 0]
    return max(0.0, float(-np.sum(pmf * np.log2(pmf + _LOG2_EPS))))


def _patch_entropy(patch: np.ndarray) -> float:
    """Shannon entropy of an RGB image patch. Mean across channels."""
    if patch.ndim != 3 or patch.shape[2] < 3:
        return 0.0
    if patch.size == 0:
        return 0.0
    r_h = _channel_entropy(patch[:, :, 0].ravel())
    g_h = _channel_entropy(patch[:, :, 1].ravel())
    b_h = _channel_entropy(patch[:, :, 2].ravel())
    return (r_h + g_h + b_h) / 3.0


# ---------------------------------------------------------------------------
# Layer 1: Observation entropy
# ---------------------------------------------------------------------------

def observation_entropy(obs: np.ndarray) -> float:
    """Shannon entropy of the full RGB image.

    Computes per-channel entropy and returns the mean. Higher values mean
    more visual complexity (more things happening in view).
    """
    return _patch_entropy(obs)


# ---------------------------------------------------------------------------
# Layer 2: Fine-grained entropy gradient
# ---------------------------------------------------------------------------

def fine_entropy_grid(obs: np.ndarray, grid_size: int = 4) -> np.ndarray:
    """Compute entropy on an NxN grid of patches.

    Returns shape (grid_size, grid_size) array of entropy values.
    Each cell is the Shannon entropy of that patch of the image.

    Default grid_size=4 gives 16 patches — 4x finer than quadrants
    while staying fast enough for real-time multi-agent use.
    """
    if obs.ndim != 3:
        return np.zeros((grid_size, grid_size))
    h, w = obs.shape[0], obs.shape[1]
    ph = max(h // grid_size, 1)
    pw = max(w // grid_size, 1)
    grid = np.zeros((grid_size, grid_size))
    for gy in range(grid_size):
        for gx in range(grid_size):
            y0 = gy * ph
            x0 = gx * pw
            y1 = min(y0 + ph, h)
            x1 = min(x0 + pw, w)
            patch = obs[y0:y1, x0:x1]
            grid[gy, gx] = _patch_entropy(patch)
    return grid


def fine_entropy_gradient(obs: np.ndarray, grid_size: int = 4) -> np.ndarray:
    """High-resolution entropy gradient from NxN grid.

    Returns (dy, dx) vector pointing from the agent (centre)
    toward the highest local entropy. Magnitude reflects signal
    strength: strong differences → magnitude near 1.0, weak/uniform
    → magnitude near 0 (noise can overcome it → natural dispersal).

    Capped at 0.5 instead of normalized — weak signals stay weak.

    Convention: dy>0 = downward, dx>0 = rightward.
    """
    grid = fine_entropy_grid(obs, grid_size)
    if grid.max() - grid.min() < 1e-6:
        return np.zeros(2)

    # Weighted centre of mass of entropy, relative to image centre
    centre = (grid_size - 1) / 2.0
    weights = grid - grid.mean()  # subtract mean so it's a gradient

    total_weight = np.abs(weights).sum()
    if total_weight < 1e-8:
        return np.zeros(2)

    # Compute weighted direction
    dy = 0.0
    dx = 0.0
    for gy in range(grid_size):
        for gx in range(grid_size):
            dy += (gy - centre) * weights[gy, gx]
            dx += (gx - centre) * weights[gy, gx]

    grad = np.array([dy, dx])
    norm = np.linalg.norm(grad)
    if norm < 1e-8:
        return np.zeros(2)
    # Unit normalize — direction preserved, magnitude capped at 1.0
    if norm > 1.0:
        return grad / norm
    return grad


def peak_local_entropy(obs: np.ndarray, grid_size: int = 4) -> float:
    """Maximum entropy across all grid patches.

    In thermodynamics, reactions happen at the hottest point — not at
    the average temperature. This returns the entropy of the most
    complex patch in the observation, which drives local interaction.
    """
    grid = fine_entropy_grid(obs, grid_size)
    return float(grid.max())


# ---------------------------------------------------------------------------
# Layer 3: Spatial change entropy (where is change happening?)
# ---------------------------------------------------------------------------

def spatial_change_grid(
    obs: np.ndarray,
    prev_obs: np.ndarray | None,
    grid_size: int = 4,
) -> np.ndarray:
    """Change entropy computed per spatial region.

    Returns (grid_size, grid_size) array where each cell is the
    Shannon entropy of the pixel-difference in that region.
    High values = lots of diverse change in that area.
    """
    if prev_obs is None or obs.shape != prev_obs.shape:
        return np.zeros((grid_size, grid_size))

    diff = np.abs(obs.astype(np.int16) - prev_obs.astype(np.int16)).astype(np.uint8)
    return fine_entropy_grid(diff, grid_size)


def spatial_change_gradient(
    obs: np.ndarray,
    prev_obs: np.ndarray | None,
    grid_size: int = 4,
) -> np.ndarray:
    """Direction of maximum change — where stuff is HAPPENING.

    Returns (dy, dx) vector pointing toward the region with
    the most diverse pixel-level change. Magnitude reflects signal
    strength: strong change → near 1.0, weak change → near 0.
    Capped at 0.5 instead of normalized — weak signals stay weak.

    Convention: dy>0 = downward, dx>0 = rightward.
    """
    grid = spatial_change_grid(obs, prev_obs, grid_size)
    if grid.max() - grid.min() < 1e-6:
        return np.zeros(2)

    centre = (grid_size - 1) / 2.0
    weights = grid - grid.mean()

    total_weight = np.abs(weights).sum()
    if total_weight < 1e-8:
        return np.zeros(2)

    dy = 0.0
    dx = 0.0
    for gy in range(grid_size):
        for gx in range(grid_size):
            dy += (gy - centre) * weights[gy, gx]
            dx += (gx - centre) * weights[gy, gx]

    grad = np.array([dy, dx])
    norm = np.linalg.norm(grad)
    if norm < 1e-8:
        return np.zeros(2)
    # Unit normalize — direction preserved, magnitude capped at 1.0
    if norm > 1.0:
        return grad / norm
    return grad


# ---------------------------------------------------------------------------
# Layer 4: Entropy rate (dS/dt)
# ---------------------------------------------------------------------------

def entropy_rate(entropy_history: list[float]) -> float:
    """Rate of entropy change over recent frames (dS/dt).

    Positive = entropy increasing (disorder growing, activity rising).
    Negative = entropy decreasing (order forming, depletion, calm).
    Zero = stable state.

    Uses linear regression slope over the history window for robustness.
    """
    if len(entropy_history) < 2:
        return 0.0
    n = len(entropy_history)
    x = np.arange(n, dtype=np.float64)
    y = np.array(entropy_history, dtype=np.float64)
    # Slope of least-squares linear fit
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom < 1e-12:
        return 0.0
    slope = ((x - x_mean) * (y - y_mean)).sum() / denom
    return float(slope)


# ---------------------------------------------------------------------------
# Layer 5: Information temperature
# ---------------------------------------------------------------------------

def information_temperature(entropy_history: list[float]) -> float:
    """How volatile the entropy signal is — thermodynamic temperature.

    High temperature = entropy is fluctuating rapidly (volatile, chaotic).
    Low temperature = entropy is stable (calm, predictable).

    T = standard deviation of entropy over recent frames.
    Returns a minimum of 0.01 to avoid division-by-zero in Boltzmann.
    """
    if len(entropy_history) < 2:
        return 1.0  # default: moderate temperature
    t = float(np.std(entropy_history))
    # Floor at 0.1 — absolute zero is unattainable in physics.
    # Prevents extreme determinism in "cold" (stable, dark) environments.
    return max(t, 0.1)


# ---------------------------------------------------------------------------
# Layer 6: Entropy production rate (thermodynamic flow)
# ---------------------------------------------------------------------------

def entropy_production_rate(entropy_history: list[float]) -> float:
    """Second derivative of entropy — is the RATE of change accelerating?

    Positive = entropy production is speeding up (system heating up).
    Negative = entropy production is slowing down (system cooling).

    This is the thermodynamic flow signal. In physical systems, entropy
    production rate determines stability of dissipative structures.
    """
    if len(entropy_history) < 3:
        return 0.0
    # First differences (velocity)
    diffs = np.diff(entropy_history)
    # Second difference (acceleration)
    accel = np.diff(diffs)
    return float(accel.mean())


# ---------------------------------------------------------------------------
# Layer 7: Spatial entropy rate (where is entropy GROWING?)
# ---------------------------------------------------------------------------

def spatial_entropy_rate(
    current_grid: np.ndarray,
    prev_grid: np.ndarray | None,
) -> np.ndarray:
    """Per-cell entropy rate: dS/dt for each spatial region.

    Positive = entropy increasing in that region (growth: apples appearing,
    agents arriving, new structure forming).
    Negative = entropy decreasing (depletion: apples eaten, agents leaving,
    pollution being cleared).

    Returns same shape as input grid.
    """
    if prev_grid is None or current_grid.shape != prev_grid.shape:
        return np.zeros_like(current_grid)
    return current_grid - prev_grid


def entropy_growth_gradient(
    current_grid: np.ndarray,
    prev_grid: np.ndarray | None,
) -> np.ndarray:
    """Direction toward where entropy is GROWING (positive dS/dt).

    Points toward apple regrowth, agent arrival, new activity.
    Away from depletion, decay, emptying regions.
    The thermodynamic arrow of time: flow toward entropy production.

    Returns (dy, dx) vector, capped at 0.5 magnitude.
    """
    rate = spatial_entropy_rate(current_grid, prev_grid)
    grid_size = rate.shape[0]

    # Only consider POSITIVE rate (growth). Zero out decay.
    growth = np.maximum(rate, 0.0)
    if growth.max() < 1e-6:
        return np.zeros(2)

    centre = (grid_size - 1) / 2.0
    weights = growth  # no mean subtraction — we want absolute growth direction

    total_weight = weights.sum()
    if total_weight < 1e-8:
        return np.zeros(2)

    dy = 0.0
    dx = 0.0
    for gy in range(grid_size):
        for gx in range(grid_size):
            dy += (gy - centre) * weights[gy, gx]
            dx += (gx - centre) * weights[gy, gx]

    grad = np.array([dy, dx])
    norm = np.linalg.norm(grad)
    if norm < 1e-8:
        return np.zeros(2)
    # Unit normalize — direction preserved, magnitude capped at 1.0
    if norm > 1.0:
        return grad / norm
    return grad


# ---------------------------------------------------------------------------
# Layer 7b: Peripheral sensing (lateral line / whiskers)
# ---------------------------------------------------------------------------

def peripheral_entropy_gradient(grid: np.ndarray) -> np.ndarray:
    """Gradient from ONLY the outermost ring of the entropy grid.

    Biological analogy: blind cave fish lateral line / mole whiskers.
    Senses what is at the PERIPHERY of the observation — a proxy for
    what lies BEYOND the observation window.

    For a 4x4 grid, the outer ring is 12 patches (all edge cells).
    The inner 2x2 is excluded. The gradient from the outer ring points
    toward the most interesting boundary of the observation.

    Accepts a pre-computed entropy grid (from fine_entropy_grid).
    Returns (dy, dx) vector, capped at 0.5 magnitude.
    """
    grid_size = grid.shape[0]

    # Outer ring mask: True for edge cells, False for interior
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    outer_vals = grid[mask]
    if outer_vals.max() - outer_vals.min() < 1e-6:
        return np.zeros(2)

    centre = (grid_size - 1) / 2.0
    outer_mean = float(outer_vals.mean())

    dy = 0.0
    dx = 0.0
    for gy in range(grid_size):
        for gx in range(grid_size):
            if mask[gy, gx]:
                w = grid[gy, gx] - outer_mean
                dy += (gy - centre) * w
                dx += (gx - centre) * w

    grad = np.array([dy, dx])
    norm = np.linalg.norm(grad)
    if norm < 1e-8:
        return np.zeros(2)
    # Unit normalize — direction preserved, magnitude capped at 1.0
    if norm > 1.0:
        return grad / norm
    return grad


# ---------------------------------------------------------------------------
# Layer 8: Thermal memory (decaying heat map)
# ---------------------------------------------------------------------------

THERMAL_DECAY = 0.85  # exponential decay per frame


def update_thermal_map(
    thermal_map: np.ndarray | None,
    current_grid: np.ndarray,
) -> np.ndarray:
    """Update the thermal memory map with adaptive cold-factor decay.

    In cold (low-entropy) environments, trails decay slower (Newton's cooling
    law — heat dissipates slower when the surroundings are cold). This helps
    agents in dark arenas (PD) maintain useful trails longer.

    At MAX_ENTROPY: effective_decay = 0.85 (unchanged from v8).
    At zero entropy: effective_decay = 0.93 (trails persist ~12 frames).
    """
    if thermal_map is None or thermal_map.shape != current_grid.shape:
        return current_grid.copy()
    mean_ent = float(current_grid.mean())
    cold_factor = max(0.0, 1.0 - mean_ent / MAX_ENTROPY)
    effective_decay = THERMAL_DECAY + 0.08 * cold_factor
    return effective_decay * thermal_map + current_grid


def thermal_gradient(thermal_map: np.ndarray) -> np.ndarray:
    """Direction toward the hottest region of the thermal memory map.

    This is heat-seeking: follow the thermal gradient toward the heat source.
    In a dark environment with moving agents, this naturally follows their
    trail and leads toward their CURRENT position.

    Returns (dy, dx) vector, capped at 0.5 magnitude.
    """
    if thermal_map.max() - thermal_map.min() < 1e-6:
        return np.zeros(2)

    grid_size = thermal_map.shape[0]
    centre = (grid_size - 1) / 2.0
    weights = thermal_map - thermal_map.mean()

    total_weight = np.abs(weights).sum()
    if total_weight < 1e-8:
        return np.zeros(2)

    dy = 0.0
    dx = 0.0
    for gy in range(grid_size):
        for gx in range(grid_size):
            dy += (gy - centre) * weights[gy, gx]
            dx += (gx - centre) * weights[gy, gx]

    grad = np.array([dy, dx])
    norm = np.linalg.norm(grad)
    if norm < 1e-8:
        return np.zeros(2)
    # Unit normalize — direction preserved, magnitude capped at 1.0
    if norm > 1.0:
        return grad / norm
    return grad


# ---------------------------------------------------------------------------
# Legacy API (change_entropy is used by action.py)
# ---------------------------------------------------------------------------

def change_entropy(obs: np.ndarray, prev_obs: np.ndarray | None) -> float:
    """Global change entropy between two frames."""
    if prev_obs is None or obs.shape != prev_obs.shape:
        return 0.0
    diff = np.abs(obs.astype(np.int16) - prev_obs.astype(np.int16)).astype(np.uint8)
    return _patch_entropy(diff)


# ---------------------------------------------------------------------------
# Layer 9: KL divergence grid (anomaly detection)
# ---------------------------------------------------------------------------

def kl_divergence_grid(obs: np.ndarray, grid_size: int = 4) -> np.ndarray:
    """KL divergence of each grid cell's pixel distribution vs the global.

    High KL = this patch looks DIFFERENT from the background.
    In a dark arena (PD), an agent's coloured sprite creates massive KL
    divergence against the uniform dark background. This amplifies
    statistically improbable patches — pure information theory, not
    object detection.

    Returns shape (grid_size, grid_size) array of KL values.
    """
    if obs.ndim != 3:
        return np.zeros((grid_size, grid_size))

    # Global pixel distribution (all channels flattened)
    global_hist, _ = np.histogram(obs.ravel(), bins=_BINS, range=(0, 256))
    global_total = global_hist.sum()
    if global_total == 0:
        return np.zeros((grid_size, grid_size))
    global_pmf = global_hist / global_total
    global_pmf = np.maximum(global_pmf, _LOG2_EPS)  # avoid log(0)

    h, w = obs.shape[0], obs.shape[1]
    ph = max(h // grid_size, 1)
    pw = max(w // grid_size, 1)
    kl_grid = np.zeros((grid_size, grid_size))

    for gy in range(grid_size):
        for gx in range(grid_size):
            y0, x0 = gy * ph, gx * pw
            y1, x1 = min(y0 + ph, h), min(x0 + pw, w)
            patch = obs[y0:y1, x0:x1]

            local_hist, _ = np.histogram(patch.ravel(), bins=_BINS, range=(0, 256))
            local_total = local_hist.sum()
            if local_total == 0:
                continue
            local_pmf = local_hist / local_total
            local_pmf = np.maximum(local_pmf, _LOG2_EPS)

            # KL(local || global) = sum(local * log2(local / global))
            kl = float(np.sum(local_pmf * np.log2(local_pmf / global_pmf)))
            kl_grid[gy, gx] = max(0.0, kl)

    return kl_grid


def kl_anomaly_gradient(obs: np.ndarray, grid_size: int = 4) -> np.ndarray:
    """Direction toward the most anomalous patch (highest KL divergence).

    Points toward statistically improbable regions. In a dark arena,
    this points toward coloured agent sprites. In a bright arena,
    this points toward unusual features (empty patches, unique objects).

    Returns (dy, dx) vector, capped at 0.5 magnitude.
    """
    kl_grid = kl_divergence_grid(obs, grid_size)
    kl_max = kl_grid.max()
    if kl_max < 1e-6:
        return np.zeros(2)

    centre = (grid_size - 1) / 2.0
    # Weight by KL value — anomalous patches pull harder
    weights = kl_grid

    total_weight = weights.sum()
    if total_weight < 1e-8:
        return np.zeros(2)

    dy = 0.0
    dx = 0.0
    for gy in range(grid_size):
        for gx in range(grid_size):
            dy += (gy - centre) * weights[gy, gx]
            dx += (gx - centre) * weights[gy, gx]

    grad = np.array([dy, dx])
    norm = np.linalg.norm(grad)
    if norm < 1e-8:
        return np.zeros(2)
    # Unit normalize — direction preserved, magnitude capped at 1.0
    if norm > 1.0:
        return grad / norm
    return grad
