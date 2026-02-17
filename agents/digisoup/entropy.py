"""Entropy computations for DigiSoup agents.

v1: Empty -- no entropy computation. Pure random baseline.

Layers will be added incrementally:
    Layer 1: Observation entropy (Shannon entropy of pixel histograms)
    Layer 2: Fine-grained entropy gradient (directional movement)
    Layer 3: Spatial change gradient (react to what's changing)
    Layer 4: Entropy rate dS/dt (is complexity growing or shrinking?)
    Layer 5: Information temperature (entropy volatility)
    Layer 6: Entropy production rate (d2S/dt2)
    Layer 7: Entropy growth gradient (where is entropy increasing?)
    Layer 8: Thermal memory (decaying heat trails)
    Layer 9: KL divergence (anomaly detection)

See ref/agent/entropy.py for the full v11 implementation.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class EntropyState(NamedTuple):
    """Immutable entropy state carried between steps.

    v1: empty. Fields will be added as layers are built.
    """
    pass


EMPTY_ENTROPY_STATE = EntropyState()


def compute_entropy_state(
    obs: np.ndarray,
    prev_state: EntropyState,
) -> EntropyState:
    """Compute the new entropy state from an observation.

    v1: no computation. Returns empty state.
    """
    return EntropyState()
