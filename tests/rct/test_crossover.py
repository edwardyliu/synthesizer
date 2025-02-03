# Test: datasynth/rct/crossover.py

import math

from datasynth.rct import CrossoverRCTGenerator


def test_crossover():
    """Test Unit: datasynth.rct.crossover"""

    treatments = {
        "discount": [0, 20],
        "time": [60, 40, 20],
    }
    n = 100
    ncopies = 3
    generator = CrossoverRCTGenerator(seed=42, ncopies=ncopies, **treatments)
    product = len(treatments["discount"]) * len(treatments["time"])
    assert len(generator.arms) == product
    assert len(generator.arm_permutations) == (math.factorial(product)) / (
        math.factorial(product - ncopies)
    )

    rct = generator.generate(n)
    assert len(rct) == ncopies * n
