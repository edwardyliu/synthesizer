# Test: datasynth/feature/normal.py

import numpy as np

from datasynth.feature import NormalDistributionGenerator


def test_normal():
    """Test Unit: datasynth.feature.normal"""

    mean = 1.0
    std = 2.0
    n = 1000
    rng = np.random.default_rng(28)

    generator = NormalDistributionGenerator(mean, std)
    feature = generator.generate(n, rng)

    assert np.isclose(np.mean(feature), mean, atol=0.1)
    assert np.isclose(np.std(feature), std, atol=0.1)
