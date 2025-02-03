# Test: datasynth/feature/t.py

import numpy as np

from datasynth.feature import TDistributionGenerator


def test_t():
    """Test Unit: datasynth.feature.t"""

    df = 10.0
    n = 1000
    rng = np.random.default_rng(28)

    generator = TDistributionGenerator(df)
    feature = generator.generate(n, rng)

    assert np.isclose(np.mean(feature), 0, atol=0.1)
