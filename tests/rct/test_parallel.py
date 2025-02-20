# Test: synthesizer/rct/parallel.py

import pandas as pd

from synthesizer.rct import ParallelRCTGenerator


def test_parallel():
    """Test Unit: synthesizer.rct.parallel"""

    treatments = {
        "discount": ["placebo", "20", "40"],
        "time": ["placebo", "40", "20"],
    }
    n = 100
    generator = ParallelRCTGenerator(seed=42, **treatments)
    assert (
        len(generator.arms) == len(treatments["discount"]) + len(treatments["time"]) - 1
    )

    rct = generator.generate(pd.DataFrame({"subject_id": range(n), "age": range(n)}))
    assert len(rct) == n
    assert len(rct.columns) == 4
