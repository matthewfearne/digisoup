"""GlyphDrift communication protocol bridge for DigiSoup (v8).

Loads evolved glyph protocols from GlyphDrift and uses them as
agent identity markers. Agents with similar protocols (same evolutionary
lineage) cooperate more readily. Different protocols reduce cooperation.

Protocol similarity is based on Jaccard overlap of top bigram vocabularies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GlyphProtocol:
    """A loaded communication protocol from GlyphDrift."""

    protocol_id: int
    top_bigrams: tuple[tuple[str, str], ...]
    compression_ratio: float

    def similarity(self, other: GlyphProtocol) -> float:
        """Jaccard similarity of top bigram sets."""
        self_set = set(self.top_bigrams)
        other_set = set(other.top_bigrams)
        if not self_set and not other_set:
            return 1.0
        if not self_set or not other_set:
            return 0.0
        intersection = len(self_set & other_set)
        union = len(self_set | other_set)
        return intersection / union


def load_glyph_protocols(path: str) -> list[GlyphProtocol]:
    """Load protocols from a GlyphDrift-exported JSON file."""
    with open(path) as f:
        data = json.load(f)
    protocols = []
    for entry in data:
        protocols.append(GlyphProtocol(
            protocol_id=entry["protocol_id"],
            top_bigrams=tuple(tuple(bg) for bg in entry["top_bigrams"]),
            compression_ratio=entry["compression_ratio"],
        ))
    return protocols


def cooperation_boost(
    agent_protocol: GlyphProtocol,
    neighbor_protocol: GlyphProtocol,
    boost_strength: float = 0.3,
) -> float:
    """Compute cooperation tendency adjustment based on protocol similarity.

    Returns a float to ADD to cooperation_tendency:
    - Positive for similar protocols (in-group bias)
    - Negative for dissimilar protocols (out-group wariness)
    - Zero when protocols are moderately similar

    The boost is scaled by boost_strength:
    - boost_strength=0.0: no protocol effect
    - boost_strength=0.3: moderate effect (default)
    - boost_strength=1.0: strong in-group bias
    """
    sim = agent_protocol.similarity(neighbor_protocol)
    # Map similarity [0, 1] to boost [-0.5, +0.5] * strength
    # 0.0 similarity -> -0.5 * strength
    # 0.5 similarity -> 0.0 (neutral)
    # 1.0 similarity -> +0.5 * strength
    return (sim - 0.5) * boost_strength


class ProtocolManager:
    """Manages protocol assignment and similarity computation for agents.

    Assign each agent a protocol_id at construction. The manager tracks
    which agents have which protocols and computes cooperation boosts
    when agents encounter each other.
    """

    def __init__(
        self,
        protocols: list[GlyphProtocol] | None = None,
        boost_strength: float = 0.3,
    ) -> None:
        self.protocols = protocols or []
        self.boost_strength = boost_strength
        self._agent_protocols: dict[int, GlyphProtocol] = {}

    @property
    def enabled(self) -> bool:
        return len(self.protocols) > 0

    def assign_protocol(self, agent_id: int, protocol_id: int) -> None:
        """Assign a protocol to an agent."""
        if not self.protocols:
            return
        proto = self.protocols[protocol_id % len(self.protocols)]
        self._agent_protocols[agent_id] = proto

    def get_protocol(self, agent_id: int) -> GlyphProtocol | None:
        """Get the protocol assigned to an agent."""
        return self._agent_protocols.get(agent_id)

    def compute_boost(self, agent_id: int, neighbor_id: int) -> float:
        """Compute cooperation boost between two agents."""
        if not self.enabled:
            return 0.0
        p1 = self._agent_protocols.get(agent_id)
        p2 = self._agent_protocols.get(neighbor_id)
        if p1 is None or p2 is None:
            return 0.0
        return cooperation_boost(p1, p2, self.boost_strength)

    def similarity_matrix(self) -> list[list[float]]:
        """Compute pairwise similarity matrix for all loaded protocols."""
        n = len(self.protocols)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix[i][j] = self.protocols[i].similarity(self.protocols[j])
        return matrix
