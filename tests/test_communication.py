"""Tests for GlyphDrift communication protocol bridge."""

import json
import os
import tempfile

import pytest

from agents.digisoup.communication import (
    GlyphProtocol,
    ProtocolManager,
    cooperation_boost,
    load_glyph_protocols,
)


@pytest.fixture
def proto_a():
    return GlyphProtocol(
        protocol_id=0,
        top_bigrams=(("a", "b"), ("b", "c"), ("c", "d")),
        compression_ratio=0.07,
    )


@pytest.fixture
def proto_b():
    return GlyphProtocol(
        protocol_id=1,
        top_bigrams=(("x", "y"), ("y", "z"), ("c", "d")),
        compression_ratio=0.08,
    )


@pytest.fixture
def proto_same_as_a():
    return GlyphProtocol(
        protocol_id=2,
        top_bigrams=(("a", "b"), ("b", "c"), ("c", "d")),
        compression_ratio=0.07,
    )


class TestGlyphProtocol:
    def test_self_similarity(self, proto_a):
        assert proto_a.similarity(proto_a) == 1.0

    def test_partial_similarity(self, proto_a, proto_b):
        sim = proto_a.similarity(proto_b)
        # 1 shared bigram out of 5 unique = 0.2
        assert sim == pytest.approx(0.2)

    def test_identical_bigrams(self, proto_a, proto_same_as_a):
        assert proto_a.similarity(proto_same_as_a) == 1.0

    def test_symmetry(self, proto_a, proto_b):
        assert proto_a.similarity(proto_b) == proto_b.similarity(proto_a)

    def test_empty(self):
        p1 = GlyphProtocol(0, (), 1.0)
        p2 = GlyphProtocol(1, (), 1.0)
        assert p1.similarity(p2) == 1.0


class TestCooperationBoost:
    def test_same_protocol_positive(self, proto_a, proto_same_as_a):
        boost = cooperation_boost(proto_a, proto_same_as_a, boost_strength=1.0)
        assert boost > 0

    def test_different_protocol_negative(self, proto_a, proto_b):
        boost = cooperation_boost(proto_a, proto_b, boost_strength=1.0)
        # Similarity is 0.2, so (0.2 - 0.5) * 1.0 = -0.3
        assert boost < 0

    def test_zero_strength(self, proto_a, proto_b):
        assert cooperation_boost(proto_a, proto_b, boost_strength=0.0) == 0.0

    def test_boost_range(self, proto_a, proto_b):
        boost = cooperation_boost(proto_a, proto_b, boost_strength=0.3)
        assert -0.15 <= boost <= 0.15


class TestProtocolManager:
    def test_assign_and_get(self, proto_a, proto_b):
        mgr = ProtocolManager([proto_a, proto_b])
        mgr.assign_protocol(0, 0)
        mgr.assign_protocol(1, 1)
        assert mgr.get_protocol(0) == proto_a
        assert mgr.get_protocol(1) == proto_b

    def test_enabled(self, proto_a):
        assert ProtocolManager([proto_a]).enabled
        assert not ProtocolManager([]).enabled
        assert not ProtocolManager(None).enabled

    def test_compute_boost_same(self, proto_a, proto_same_as_a):
        mgr = ProtocolManager([proto_a, proto_same_as_a], boost_strength=1.0)
        mgr.assign_protocol(0, 0)
        mgr.assign_protocol(1, 1)
        boost = mgr.compute_boost(0, 1)
        assert boost > 0

    def test_compute_boost_disabled(self):
        mgr = ProtocolManager([])
        assert mgr.compute_boost(0, 1) == 0.0

    def test_similarity_matrix(self, proto_a, proto_b):
        mgr = ProtocolManager([proto_a, proto_b])
        matrix = mgr.similarity_matrix()
        assert len(matrix) == 2
        assert matrix[0][0] == 1.0
        assert matrix[1][1] == 1.0
        assert matrix[0][1] == matrix[1][0]

    def test_protocol_wraps(self, proto_a, proto_b):
        """Protocol assignment wraps around for IDs larger than protocol count."""
        mgr = ProtocolManager([proto_a, proto_b])
        mgr.assign_protocol(0, 4)  # 4 % 2 = 0
        assert mgr.get_protocol(0) == proto_a


class TestLoadProtocols:
    def test_load_roundtrip(self, proto_a, proto_b):
        data = [
            {
                "protocol_id": proto_a.protocol_id,
                "top_bigrams": [list(bg) for bg in proto_a.top_bigrams],
                "compression_ratio": proto_a.compression_ratio,
            },
            {
                "protocol_id": proto_b.protocol_id,
                "top_bigrams": [list(bg) for bg in proto_b.top_bigrams],
                "compression_ratio": proto_b.compression_ratio,
            },
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        try:
            loaded = load_glyph_protocols(path)
            assert len(loaded) == 2
            assert loaded[0].protocol_id == 0
            assert loaded[1].top_bigrams == proto_b.top_bigrams
        finally:
            os.unlink(path)

    def test_load_real_protocols(self):
        """Load the actual generated protocols if they exist."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "..", "glyphdrift", "protocols.json",
        )
        # This test is a smoke test — skip if file doesn't exist
        if not os.path.exists(path):
            pytest.skip("protocols.json not found")
        protocols = load_glyph_protocols(path)
        assert len(protocols) >= 2
