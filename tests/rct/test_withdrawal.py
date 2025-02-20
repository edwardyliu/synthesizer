# Test: synthesizer/rct/withdrawal.py

import pytest
import pandas as pd

from synthesizer.rct import WithdrawalRCTGenerator


def test_withdrawal():
    """Test Unit: synthesizer.rct.withdrawal"""

    treatments = {
        "discount": ["placebo", "20"],
        "time": ["placebo", "40"],
    }
    n = 1000
    generator = WithdrawalRCTGenerator(
        seed=42, response=0.5, withdrawal=0.5, **treatments
    )
    assert (
        len(generator.arms) == len(treatments["discount"]) + len(treatments["time"]) - 1
    )

    rct = generator.generate(pd.DataFrame({"subject_id": range(n), "age": range(n)}))
    assert len(rct) == n

    # 1 for sid
    # 2 columns for response and status
    # 2 columns for treatment
    # 1 column for feature "age"
    assert len(rct.columns) == 1 + 2 + 2 + 1
    assert sum(rct["response"] == True) / n == pytest.approx(0.5, abs=0.1)
    assert sum(rct["status"] == True) / n == pytest.approx(0.25, abs=0.1)
    assert sum(rct["status"] == False) / n == pytest.approx(0.25, abs=0.1)
