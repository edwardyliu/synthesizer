# Test: datasynth/rct/crossover.py

import math

from datasynth.rct import CrossoverRCTGenerator


def test_crossover():
    """Test Unit: datasynth.rct.crossover"""

    treatments = {
        "discount": ["placebo", "20"],
        "time": ["placebo", "40", "20"],
    }
    n = 100
    ncopies = 2
    generator = CrossoverRCTGenerator(seed=42, ncopies=ncopies, **treatments)
    product = len(treatments["discount"]) * len(treatments["time"])
    assert len(generator.arms) == product

    rct = generator.generate(n)
    assert len(rct) == ncopies * n
