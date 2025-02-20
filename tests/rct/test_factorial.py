# Test: synthesizer/rct/factorial.py

import pandas as pd

from synthesizer.rct import FactorialRCTGenerator


def test_factorial():
    """Test Unit: synthesizer.rct.factorial"""

    treatments = {
        "discount": ["placebo", "20", "40"],
        "time": ["placebo", "40", "20"],
    }
    n = 100
    generator = FactorialRCTGenerator(seed=42, **treatments)
    assert len(generator.arms) == len(treatments["discount"]) * len(treatments["time"])

    rct = generator.generate(pd.DataFrame({"subject_id": range(n), "age": range(n)}))
    assert len(rct) == n
    assert len(rct.columns) == 4
