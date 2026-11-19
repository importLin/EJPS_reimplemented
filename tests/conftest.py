"""Shared fixtures: the golden values in ``tests/fixtures`` and their input recipes."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))  # make_fixtures: deterministic input recipes

FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def golden() -> dict:
    return json.loads((FIXTURES / "golden.json").read_text())


@pytest.fixture(scope="session")
def golden_arrays():
    with np.load(FIXTURES / "golden.npz") as data:
        return {k: data[k] for k in data.files}
