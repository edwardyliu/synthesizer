# Test: synthesizer/feature/geometric.py

import pytest

import numpy as np

from synthesizer.feature import GeometricDistributionGenerator


def test_geometric():
    """Test Unit: synthesizer.feature.choice"""

    success = 0.5
    n = 1000
    rng = np.random.default_rng(28)

    generator = GeometricDistributionGenerator(success)
    feature = generator.generate(n, rng)

    # how many trials succeeded after a single run?
    assert sum(feature == 1) / n == pytest.approx(0.5, abs=0.1)
