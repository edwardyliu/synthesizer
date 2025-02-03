# Test: datasynth/rct/parallel.py

from datasynth.rct import ParallelRCTGenerator


def test_parallel():
    """Test Unit: datasynth.rct.parallel"""

    treatments = {
        "discount": ["placebo", "20", "40"],
        "time": ["placebo", "40", "20"],
    }
    n = 100
    generator = ParallelRCTGenerator(seed=42, **treatments)
    assert (
        len(generator.arms) == len(treatments["discount"]) + len(treatments["time"]) - 1
    )

    rct = generator.generate(n)
    assert len(rct) == n
