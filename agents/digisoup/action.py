"""Action selection for DigiSoup agents.

v1: Pure uniform random. The statistical floor.

Action selection will become entropy-driven as layers are added.
See ref/agent/action.py for the full v11 implementation.

Melting Pot action space:
    0: noop
    1: forward
    2: backward
    3: strafe left
    4: strafe right
    5: turn left
    6: turn right
    7: interact (zap / cooperate / etc.)
    8: fire_clean (Clean Up only -- 9-action substrate)
"""
from __future__ import annotations

import numpy as np

from agents.digisoup.entropy import EntropyState


def select_action(
    entropy_state: EntropyState,
    n_actions: int = 8,
    step_count: int = 0,
    rng: np.random.Generator | None = None,
) -> int:
    """Select an action.

    v1: uniform random over all available actions.
    """
    rng = rng or np.random.default_rng()
    return int(rng.integers(0, n_actions))
