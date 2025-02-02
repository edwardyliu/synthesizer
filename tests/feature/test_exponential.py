# Test: datasynth/feature/exponential.py

import numpy as np

from datasynth.feature import ExponentialDistributionGenerator


def test_exponential():
    """Test Unit: datasynth.feature.exponential"""

    scale = 1.0
    n = 1000
    rng = np.random.default_rng(28)

    generator = ExponentialDistributionGenerator(scale)
    feature = generator.generate(n, rng)

    assert np.isclose(np.min(feature), 0, atol=0.1)
