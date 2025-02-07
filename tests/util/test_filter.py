# Test: synthesizer/util/test_filter.py

from synthesizer import DatasetSynthesizer
from synthesizer.feature import (
    ChoiceDistributionGenerator,
    NormalDistributionGenerator,
    UniformIntegerDistributionGenerator,
)
from synthesizer.util import Filters, Filter


def test_synthesizer():
    """Test Unit: synthesizer.synthesizer"""

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

    n = 10000
    dataset = synthesizer.generate(n)
    filters = Filters(
        [
            Filter("age", 18, "__ge__"),
            Filter("income", 55000, "__le__"),
            Filter("occupation", "engineering"),
        ]
    )
    dataset_f = filters(dataset)
    assert dataset_f["age"].min() >= 18
    assert dataset_f["income"].max() <= 55000
    assert dataset_f["occupation"].unique() == "engineering"
