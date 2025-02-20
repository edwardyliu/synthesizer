# Test: synthesizer/blocking/generator.py

import pandas as pd

from synthesizer import DatasetSynthesizer
from synthesizer.feature import (
    ChoiceDistributionGenerator,
    NormalDistributionGenerator,
    UniformIntegerDistributionGenerator,
)
from synthesizer.blocking import SimpleBlockingGenerator
from synthesizer.util import Filters, Filter


def test_simple_generator():
    """Test Unit: synthesizer.blocking.SimpleBlockingGenerator"""

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

    generator = SimpleBlockingGenerator()
    blocked = generator.generate(
        subjects,
        {
            "BLOCKING_1": Filters(
                [
                    Filter("gender", "male"),
                    Filter("age", 18, "__ge__"),
                    Filter("age", 25, "__le__"),
                ]
            ),
            "BLOCKING_2": Filters(
                [
                    Filter("gender", "female"),
                    Filter("age", 18, "__ge__"),
                    Filter("age", 25, "__le__"),
                ]
            ),
        },
    )
    assert len(subjects.columns) + 1 == len(blocked.columns)
    assert "block" not in subjects.columns
    assert "block" in blocked.columns
    assert blocked[blocked["gender"] == "male"]["block"].unique() == "BLOCKING_1"
    assert blocked[blocked["gender"] == "male"]["age"].min() >= 18
    assert blocked[blocked["gender"] == "male"]["age"].max() >= 25
    assert blocked[blocked["gender"] == "female"]["block"].unique() == "BLOCKING_2"
    assert blocked[blocked["gender"] == "female"]["age"].min() >= 18
    assert blocked[blocked["gender"] == "female"]["age"].max() >= 25
