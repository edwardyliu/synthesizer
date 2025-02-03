# Test: synthesizer/feature/poisson.py

import numpy as np

from synthesizer.feature import PoissonDistributionGenerator


def test_poisson():
    """Test Unit: synthesizer.feature.choice"""

    events = 5
    n = 1000
    rng = np.random.default_rng(28)

    generator = PoissonDistributionGenerator(events)
    feature = generator.generate(n, rng)

    assert np.isclose(np.mean(feature), 5, atol=0.1)
