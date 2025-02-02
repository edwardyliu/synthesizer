# Test: datasynth/rct/parallel.py

import numpy as np

from datasynth.rct import ParallelGenerator


def test_parallel():
    """Test Unit: datasynth.rct.parallel"""

    treatments = {
        "discount": [0, 20, 40],
        "time": [60, 40, 20],
    }
    n = 100
    rng = np.random.default_rng(28)

    generator = ParallelGenerator(**treatments)
    assert len(generator.arms) == len(treatments["discount"]) * len(treatments["time"])

    rct = generator.generate(n, rng)
    assert len(rct) == n
