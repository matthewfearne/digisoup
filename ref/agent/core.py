"""DigiSoup agent for Melting Pot — v8 (9-Layer Entropy Stack).

The agent knows ONE thing: entropy — and its thermodynamics.

9-layer entropy stack:
  1. Observation entropy          5. Information temperature
  2. Fine entropy gradient        6. Entropy production rate
  3. Spatial change gradient      7. Entropy growth gradient
  4. Entropy rate (dS/dt)         8. Thermal memory trails
                                  9. KL divergence (anomaly)

From the original DigiSoup: "You do not need to make agents cooperate.
You need to create thermodynamic conditions where cooperation dissipates
more entropy than defection. Then cooperation becomes inevitable."

NO perception of agents or resources. NO energy model from the game.
NO cooperation tracking. NO reward optimization. NO training. NO rules.

The environment speaks through entropy. Thermodynamics IS the policy.
"""
from __future__ import annotations

import numpy as np

from .action import select_action
from .entropy import (
    fine_entropy_grid,
    fine_entropy_gradient,
    observation_entropy,
    spatial_change_gradient,
    update_thermal_map,
)


# How many frames of entropy history to keep for thermodynamic calculations.
_HISTORY_WINDOW = 20

# How many gradient vectors to keep for momentum calculation.
_GRADIENT_WINDOW = 10


class DigiSoupAgent:
    """Zero-training, thermodynamic entropy-driven agent."""

    def __init__(
        self,
        agent_id: int = 0,
        seed: int | None = None,
        n_actions: int = 8,
        personality: dict | None = None,
    ) -> None:
        self.agent_id = agent_id
        self._n_actions = n_actions
        self._personality = personality or {
            "interact_bias": 1.0, "move_bias": 1.0, "scan_bias": 1.0,
        }
        self._rng = np.random.default_rng(seed)
        self._step_count: int = 0
        self._last_entropy: float = 0.0
        self._prev_obs: np.ndarray | None = None
        self._entropy_history: list[float] = []
        self._gradient_history: list[np.ndarray] = []
        # Layer 7: previous entropy grid for spatial dS/dt
        self._prev_grid: np.ndarray | None = None
        # Layer 8: thermal memory map (decaying heat map)
        self._thermal_map: np.ndarray | None = None
        # Echo response (echolocation): track action-entropy coupling
        self._prev_action: int = 0
        self._echo_strength: float = 0.0
        self._ambient_strength: float = 0.0
        # Echo chase: direction of last echo (WHERE the reactive thing was)
        self._echo_direction: np.ndarray = np.zeros(2)
        # Echo trend: is echo_resonance rising or falling? (migration signal)
        self._echo_history: list[float] = []

    def act(self, observation: np.ndarray, reward: float = 0.0) -> int:
        """Select an action from the observation. Thermodynamic entropy only.

        reward is accepted for API compatibility but NEVER used.
        """
        # Handle dict observations from Melting Pot / PettingZoo.
        if isinstance(observation, dict):
            observation = observation.get(
                "RGB", observation.get("rgb",
                    np.zeros((88, 88, 3), dtype=np.uint8)))

        observation = np.asarray(observation, dtype=np.uint8)
        if observation.ndim == 2:
            observation = np.stack([observation] * 3, axis=-1)

        # Layer 1: current observation entropy
        self._last_entropy = observation_entropy(observation)

        # Echo response: track entropy change after INTERACT vs other actions
        _interact_action = self._n_actions - 1
        _alpha = 0.15
        if self._entropy_history:
            _prev_ent = self._entropy_history[-1]
            _delta_ent = abs(self._last_entropy - _prev_ent)
            if self._prev_action == _interact_action:
                self._echo_strength = (1 - _alpha) * self._echo_strength + _alpha * _delta_ent
            else:
                self._ambient_strength = (1 - _alpha) * self._ambient_strength + _alpha * _delta_ent
        echo_resonance = self._echo_strength / max(self._ambient_strength, 0.01)

        # Echo direction: WHERE did the echo come from?
        # After INTERACT, the spatial change gradient points toward the change source
        if self._prev_action == _interact_action and self._prev_obs is not None:
            echo_ch_grad = spatial_change_gradient(observation, self._prev_obs, grid_size=4)
            echo_ch_norm = float(np.linalg.norm(echo_ch_grad))
            if echo_ch_norm > 0.02:
                self._echo_direction = echo_ch_grad
        else:
            # Decay echo direction when NOT interacting
            self._echo_direction = self._echo_direction * 0.9

        # Echo trend: is echo_resonance rising or falling?
        # Declining echo = diminishing returns = zone exhausted = migrate
        self._echo_history.append(echo_resonance)
        if len(self._echo_history) > 10:
            self._echo_history = self._echo_history[-10:]
        echo_trend = 0.0
        if len(self._echo_history) >= 5:
            recent = float(np.mean(self._echo_history[-3:]))
            older = float(np.mean(self._echo_history[:3]))
            echo_trend = recent - older

        # Track entropy history for thermodynamic layers (4, 5, 6)
        self._entropy_history.append(self._last_entropy)
        if len(self._entropy_history) > _HISTORY_WINDOW:
            self._entropy_history = self._entropy_history[-_HISTORY_WINDOW:]

        # Compute current entropy grid (used for Layer 7 + Layer 8)
        current_grid = fine_entropy_grid(observation, grid_size=4)

        # Layer 8: update thermal memory map
        self._thermal_map = update_thermal_map(
            self._thermal_map, current_grid
        )

        # Select action using full 9-layer thermodynamic stack + echo
        action = select_action(
            obs=observation,
            prev_obs=self._prev_obs,
            entropy_history=self._entropy_history,
            gradient_history=self._gradient_history,
            prev_grid=self._prev_grid,
            thermal_map=self._thermal_map,
            echo_resonance=echo_resonance,
            echo_direction=self._echo_direction,
            echo_trend=echo_trend,
            n_actions=self._n_actions,
            personality=self._personality,
            step_count=self._step_count,
            rng=self._rng,
        )

        # Track combined gradient for momentum calculation
        grad = fine_entropy_gradient(observation, grid_size=4)
        ch_grad = spatial_change_gradient(observation, self._prev_obs, grid_size=4)
        ch_strength = float(np.linalg.norm(ch_grad))
        combined = (grad + ch_grad) / 2.0 if ch_strength > 0.05 else grad
        self._gradient_history.append(combined)
        if len(self._gradient_history) > _GRADIENT_WINDOW:
            self._gradient_history = self._gradient_history[-_GRADIENT_WINDOW:]

        # Update state for next frame
        self._prev_action = action
        self._prev_obs = observation.copy()
        self._prev_grid = current_grid
        self._step_count += 1

        return action

    def reset(self) -> None:
        """Reset for a new episode."""
        self._step_count = 0
        self._last_entropy = 0.0
        self._prev_obs = None
        self._entropy_history = []
        self._gradient_history = []
        self._prev_grid = None
        self._thermal_map = None
        self._prev_action = 0
        self._echo_strength = 0.0
        self._ambient_strength = 0.0
        self._echo_direction = np.zeros(2)
        self._echo_history = []

    def summary(self) -> dict:
        """Snapshot for logging."""
        from .entropy import information_temperature, entropy_rate
        return {
            "agent_id": self.agent_id,
            "step": self._step_count,
            "entropy": self._last_entropy,
            "temperature": information_temperature(self._entropy_history),
            "dS_dt": entropy_rate(self._entropy_history),
        }

    def __repr__(self) -> str:
        from .entropy import information_temperature
        T = information_temperature(self._entropy_history)
        return (
            f"DigiSoupAgent(id={self.agent_id}, "
            f"entropy={self._last_entropy:.2f}, "
            f"T={T:.3f}, "
            f"step={self._step_count})"
        )
