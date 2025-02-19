# Test: synthesizer/dataset/test_synthesizer.py

from synthesizer import DatasetSynthesizer
from synthesizer.feature import (
    ChoiceDistributionGenerator,
    NormalDistributionGenerator,
    UniformIntegerDistributionGenerator,
)


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

    n = 100
    dataset = synthesizer.generate(n)
    assert dataset["subject_id"].min() == 1
    assert len(dataset) == n
    assert len(dataset.columns) == 5
    assert "subject_id" in dataset.columns
    assert "copy" in dataset.columns
    assert "age" in dataset.columns
    assert "income" in dataset.columns
    assert "occupation" in dataset.columns
    assert dataset["age"].min() >= 12
    assert dataset["age"].max() <= 65

    dataset = synthesizer.generate(n, sid_start=n)
    assert dataset["subject_id"].min() == n + 1
    assert dataset["subject_id"].max() == n + n


def test_synthesizer_duplicates():
    """Test Unit: synthesizer.synthesizer"""

    synthesizer = DatasetSynthesizer(28)
    synthesizer.add_feature(
        "age_t1",
        UniformIntegerDistributionGenerator(18, 65),
        duplicates=["age_t2"],
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
        duplicates=["occupation_t2", "occupation_t3"],
    )

    n = 100
    dataset = synthesizer.generate(n)
    assert dataset["subject_id"].min() == 1
    assert len(dataset) == n
    assert len(dataset.columns) == 8
    assert "subject_id" in dataset.columns
    assert "copy" in dataset.columns
    assert "age_t1" in dataset.columns
    assert "age_t2" in dataset.columns
    assert sum(dataset["age_t1"] == dataset["age_t2"]) == n
    assert "income" in dataset.columns
    assert "occupation_t1" in dataset.columns
    assert "occupation_t2" in dataset.columns
    assert "occupation_t3" in dataset.columns
    assert sum(dataset["occupation_t1"] == dataset["occupation_t2"]) == n
    assert sum(dataset["occupation_t2"] == dataset["occupation_t3"]) == n

    dataset = synthesizer.generate(n, sid_start=n)
    assert dataset["subject_id"].min() == n + 1
    assert dataset["subject_id"].max() == n + n


def test_synthesizer_copies():
    """Test Unit: synthesizer.synthesizer"""

    static = ["age", "occupation"]
    synthesizer = DatasetSynthesizer(28, static)
    synthesizer.add_feature("age", UniformIntegerDistributionGenerator(18, 65))
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
        "score",
        NormalDistributionGenerator(60, 1.5),
    )
    synthesizer.add_feature(
        "income",
        NormalDistributionGenerator(65000, 1.0, decimals=2),
    )

    n = 100
    ncopies = 3
    dataset = synthesizer.generate(n, ncopies=ncopies)
    assert len(dataset) == ncopies * n
    assert len(dataset.columns) == 6
    assert "subject_id" in dataset.columns
    assert "copy" in dataset.columns
    assert "age" in dataset.columns
    assert "occupation" in dataset.columns
    assert "score" in dataset.columns
    assert "income" in dataset.columns
