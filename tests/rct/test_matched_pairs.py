# Test: synthesizer/rct/matched_pairs.py

import pandas as pd

from synthesizer import DatasetSynthesizer
from synthesizer.feature import (
    ChoiceDistributionGenerator,
    NormalDistributionGenerator,
    UniformIntegerDistributionGenerator,
)
from synthesizer.rct import MatchedPairsRCTGenerator
from synthesizer.util import Filters, Filter


def test_matched_pairs():
    """Test Unit: synthesizer.rct.matched_pairs"""

    synthesizer = DatasetSynthesizer(28)
    synthesizer.add_feature(
        "age",
        [
            UniformIntegerDistributionGenerator(18, 65),
            UniformIntegerDistributionGenerator(12, 16),
        ],
    )
    synthesizer.add_feature(
        "income",
        [
            NormalDistributionGenerator(65000, 10000, decimals=2),
            NormalDistributionGenerator(120000, 30000, decimals=2),
            NormalDistributionGenerator(280000, 50000, decimals=2),
        ],
    )
    synthesizer.add_feature(
        "occupation",
        ChoiceDistributionGenerator(
            categories=[
                "engineering",
                "medical",
                "education",
                "student",
            ]
        ),
    )
    synthesizer.add_feature(
        "gender",
        ChoiceDistributionGenerator(
            categories=[
                "male",
                "female",
                "other",
            ]
        ),
    )
    n = 1000
    subjects = synthesizer.generate(n)

    treatments = {
        "discount": ["placebo", "20"],
        "time": ["placebo", "40"],
    }
    generator = MatchedPairsRCTGenerator(seed=42, **treatments)
    experiment = generator.generate(
        subjects,
        {
            "GROUP_1": Filters([Filter("gender", "male"), Filter("age", 18, "__ge__")]),
            "GROUP_2": Filters(
                [Filter("gender", "female"), Filter("age", 18, "__ge__")]
            ),
        },
    )
    # 4 columns for features
    # 1 for copy
    # 2 for treatments
    # 1 for group
    # 1 for sid
    assert len(experiment.columns) == 4 + 1 + 2 + 1 + 1
    assert experiment[experiment["gender"] == "male"]["group"].unique() == "GROUP_1"
    assert experiment[experiment["group"] == "GROUP_1"]["age"].min() >= 18
    assert experiment[experiment["gender"] == "female"]["group"].unique() == "GROUP_2"
    assert experiment[experiment["group"] == "GROUP_2"]["age"].min() >= 18
