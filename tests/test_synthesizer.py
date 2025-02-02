# Test: datasynth/test_synthesizer.py

from datasynth import DataSynthesizer
from datasynth.feature import (
    ChoiceDistributionGenerator,
    NormalDistributionGenerator,
    UniformIntegerDistributionGenerator,
)


def test_synthesizer():
    """Test Unit: datasynth.synthesizer"""

    synthesizer = DataSynthesizer(28)
    synthesizer.add_feature("age", UniformIntegerDistributionGenerator(18, 65))
    synthesizer.add_feature("income", NormalDistributionGenerator(65000, 1.0, round=2))
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

    dataset = synthesizer.generate(100)
    assert len(dataset) == 100
    assert len(dataset.columns) == 4
    assert "subject_id" in dataset.columns
    assert "age" in dataset.columns
    assert "income" in dataset.columns
    assert "occupation" in dataset.columns
