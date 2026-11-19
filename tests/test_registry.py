"""Tests for ejps/registry.py and the public API."""

import pytest

import ejps
from ejps.registry import Registry


def test_register_build_and_lookup():
    reg = Registry("thing")

    @reg.register("double")
    def double(x, factor=2):
        return x * factor

    @reg.register()
    class Box:
        def __init__(self, v):
            self.v = v

    assert reg.build("double", 3) == 6 and reg.build("double", 3, factor=5) == 15
    assert reg.build("Box", 7).v == 7
    assert reg.get("double") is double
    assert reg.names() == ["Box", "double"] and list(reg) == ["Box", "double"]
    assert "Box" in reg and "nope" not in reg and len(reg) == 2
    assert repr(reg) == "Registry('thing', ['Box', 'double'])"
    with pytest.raises(KeyError, match="already registered"):
        reg.register("double")(double)
    with pytest.raises(KeyError, match="unknown 'nope'; available"):
        reg.get("nope")
    with pytest.raises(ValueError, match="empty name"):
        reg.register("")(double)


def test_public_api():
    assert ejps.EJPS_SOLVERS.names() == ["edge_ref", "layer_greedy"]
    assert ejps.KEY_GROUP_KINDS == ("common", "independent", "per_position")
    assert isinstance(ejps.EJPS_SOLVERS.build("layer_greedy"), ejps.LayerGreedySolver)
    assert set(ejps.__all__) <= set(dir(ejps))
