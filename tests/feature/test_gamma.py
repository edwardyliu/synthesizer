# Test: datasynth/feature/gamma.py

import numpy as np

from datasynth.feature import GammaDistributionGenerator


def test_gamma():
    """Test Unit: datasynth.feature.gamma"""

    shape = 2.0
    scale = 2.0
    n = 1000
    rng = np.random.default_rng(28)

    generator = GammaDistributionGenerator(shape, scale)
    feature = generator.generate(n, rng)

    assert np.isclose(np.mean(feature), 4, atol=0.1)
    assert np.isclose(np.std(feature), 2.8, atol=0.1)
