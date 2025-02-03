# Test: datasynth/feature/poisson.py

import numpy as np

from datasynth.feature import PoissonDistributionGenerator


def test_poisson():
    """Test Unit: datasynth.feature.choice"""

    events = 5
    n = 1000
    rng = np.random.default_rng(28)

    generator = PoissonDistributionGenerator(events)
    feature = generator.generate(n, rng)

    assert np.isclose(np.mean(feature), 5, atol=0.1)
