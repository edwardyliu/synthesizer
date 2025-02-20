# Test: synthesizer/rct/crossover.py

import pandas as pd

from synthesizer.rct import CrossoverRCTGenerator


def test_crossover():
    """Test Unit: synthesizer.rct.crossover"""

    treatments = {
        "discount": ["placebo", "20"],
        "time": ["placebo", "40", "20"],
    }
    n = 100
    ncopies = 2
    generator = CrossoverRCTGenerator(seed=42, ncopies=ncopies, **treatments)
    product = len(treatments["discount"]) * len(treatments["time"])
    assert len(generator.arms) == product

    rct = generator.generate(pd.DataFrame({"subject_id": range(n)}))
    assert len(rct) == ncopies * n
