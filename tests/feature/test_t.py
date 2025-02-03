# Test: synthesizer/feature/t.py

import numpy as np

from synthesizer.feature import TDistributionGenerator


def test_t():
    """Test Unit: synthesizer.feature.t"""

    df = 10.0
    n = 1000
    rng = np.random.default_rng(28)

    generator = TDistributionGenerator(df)
    feature = generator.generate(n, rng)

    assert np.isclose(np.mean(feature), 0, atol=0.1)
