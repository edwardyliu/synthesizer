# Test: synthesizer/rct/withdrawal.py

import pytest

from synthesizer.rct import WithdrawalRCTGenerator


def test_withdrawal():
    """Test Unit: synthesizer.rct.withdrawal"""

    treatments = {
        "discount": ["placebo", "20"],
        "time": ["placebo", "40"],
    }
    n = 1000
    generator = WithdrawalRCTGenerator(
        seed=42, success=0.5, withdrawal=0.5, **treatments
    )
    assert (
        len(generator.arms) == len(treatments["discount"]) + len(treatments["time"]) - 1
    )

    rct = generator.generate(n)
    assert len(rct) == n
    assert rct["subject_id"].min() == 1
    assert sum(rct["response"] == True) / n == pytest.approx(0.5, abs=0.1)
    assert sum(rct["withdrawal"] == True) / n == pytest.approx(0.25, abs=0.1)
    assert sum(rct["withdrawal"] == False) / n == pytest.approx(0.25, abs=0.1)

    rct = generator.generate(n, sid_start=n)
    assert rct["subject_id"].min() == n + 1
    assert rct["subject_id"].max() == n + n
