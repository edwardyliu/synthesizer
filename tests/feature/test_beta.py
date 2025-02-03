# Test: synthesizer/feature/beta.py

import pytest

import numpy as np

from synthesizer.feature import BetaDistributionGenerator


def test_beta():
    """Test Unit: synthesizer.feature.choice"""

    a = 2.0
    b = 2.0
    n = 1000
    rng = np.random.default_rng(28)

    generator = BetaDistributionGenerator(a, b)
    feature = generator.generate(n, rng)

    assert feature.mean() == pytest.approx(0.5, abs=0.1)
