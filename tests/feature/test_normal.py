# Test: synthesizer/feature/normal.py

import numpy as np

from synthesizer.feature import NormalDistributionGenerator


def test_normal():
    """Test Unit: synthesizer.feature.normal"""

    mean = 1.0
    std = 2.0
    n = 1000
    rng = np.random.default_rng(28)

    generator = NormalDistributionGenerator(mean, std)
    feature = generator.generate(n, rng)

    assert np.isclose(np.mean(feature), mean, atol=0.1)
    assert np.isclose(np.std(feature), std, atol=0.1)
    assert np.min(feature) < 0

    generator = NormalDistributionGenerator(mean, std, min=0)
    feature = generator.generate(n, rng)
    assert np.min(feature) == 0
