# Test: synthesizer/feature/uniform.py

import numpy as np

from synthesizer.feature import UniformDistributionGenerator


def test_uniform():
    """Test Unit: synthesizer.feature.uniform"""

    low = 0.0
    high = 1.0
    n = 1000
    rng = np.random.default_rng(28)

    generator = UniformDistributionGenerator(low, high)
    feature = generator.generate(n, rng)

    assert np.isclose(np.min(feature), low, atol=0.1)
    assert np.isclose(np.max(feature), high, atol=0.1)
