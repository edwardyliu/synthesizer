# Test: synthesizer/feature/logistic.py

import numpy as np

from synthesizer.feature import LogisticDistributionGenerator


def test_logistic():
    """Test Unit: synthesizer.feature.logistic"""

    mean = 10.0
    scale = 1.0
    n = 10000
    rng = np.random.default_rng(28)

    generator = LogisticDistributionGenerator(mean, scale)
    feature = generator.generate(n, rng)

    assert np.isclose(np.mean(feature), mean, atol=0.2)
