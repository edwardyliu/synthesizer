# Test: synthesizer/feature/binomial.py

import pytest

import numpy as np

from synthesizer.feature import BinomialDistributionGenerator


def test_binomial():
    """Test Unit: synthesizer.feature.choice"""

    trials = 1
    success = 0.5
    n = 1000
    rng = np.random.default_rng(28)

    generator = BinomialDistributionGenerator(trials, success)
    feature = generator.generate(n, rng)

    # flipping a coin 1 time, tested 1000 times; half should be heads, other half tails
    assert sum(feature == 0) / n == pytest.approx(0.5, abs=0.1)
    assert sum(feature == 1) / n == pytest.approx(0.5, abs=0.1)
