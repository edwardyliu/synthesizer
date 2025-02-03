# Test: synthesizer/feature/f.py

import numpy as np

from synthesizer.feature import FDistributionGenerator


def test_f():
    """Test Unit: synthesizer.feature.f"""

    dfnum = 1.0
    dfden = 48.0
    n = 1000
    rng = np.random.default_rng(28)

    generator = FDistributionGenerator(dfnum, dfden)
    feature = generator.generate(n, rng)

    assert np.isclose(np.mean(feature), 1, atol=0.1)
