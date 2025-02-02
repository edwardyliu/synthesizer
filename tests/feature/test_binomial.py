# Test: datasynth/feature/binomial.py

import pytest

import numpy as np

from datasynth.feature import BinomialDistributionGenerator


def test_choice():
    """Test Unit: datasynth.feature.choice"""

    trials = 1
    success = 0.5
    n = 1000
    rng = np.random.default_rng(28)

    generator = BinomialDistributionGenerator(trials, success)
    feature = generator.generate(n, rng)

    assert sum(feature == 0) / n == pytest.approx(0.5, abs=0.1)
