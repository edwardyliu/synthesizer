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
    synthesizer.add_feature(
        "age_t1",
        UniformIntegerDistributionGenerator(18, 65),
        duplicate=["age_t2"],
    )
    synthesizer.add_feature(
        "income", NormalDistributionGenerator(65000, 1.0, decimals=2)
    )
    synthesizer.add_feature(
        "occupation_t1",
        ChoiceDistributionGenerator(
            categories=[
                "engineering",
                "medical",
                "education",
                "student",
            ]
        ),
        duplicate=["occupation_t2", "occupation_t3"],
    )

    n = 100
    dataset = synthesizer.generate(n)
    assert len(dataset) == n
    assert len(dataset.columns) == 7
    assert "subject_id" in dataset.columns
    assert "age_t1" in dataset.columns
    assert "age_t2" in dataset.columns
    assert sum(dataset["age_t1"] == dataset["age_t2"]) == n
    assert "income" in dataset.columns
    assert "occupation_t1" in dataset.columns
    assert "occupation_t2" in dataset.columns
    assert "occupation_t3" in dataset.columns
    assert sum(dataset["occupation_t1"] == dataset["occupation_t2"]) == n
    assert sum(dataset["occupation_t2"] == dataset["occupation_t3"]) == n
