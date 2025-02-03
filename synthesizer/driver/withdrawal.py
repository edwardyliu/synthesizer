# synthesizer/driver/withdrawal.py

import logging

logger = logging.getLogger("synthesizer")


from synthesizer import DatasetSynthesizer
from synthesizer.feature import (
    ChoiceDistributionGenerator,
    NormalDistributionGenerator,
    UniformIntegerDistributionGenerator,
)
from synthesizer.rct import WithdrawalRCTGenerator


def main():
    logger.info("synthesizer.driver.withdrawal.main: generating dataset")

    # build data synthesizer
    synthesizer = DatasetSynthesizer()
    synthesizer.add_feature("age", UniformIntegerDistributionGenerator(18, 65))
    synthesizer.add_feature(
        "income", NormalDistributionGenerator(65000, 1.0, decimals=2)
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

    # design and generate the RCT
    treatments = {
        "discount": ["placebo", "20", "40"],
        "time": ["placebo", "40", "20"],
    }
    rct = WithdrawalRCTGenerator(**treatments)
    rct.arms = [
        {"discount": "20", "time": "placebo"},
        {"discount": "40", "time": "placebo"},
    ]

    # generate dataset and design for <n> subjects
    n = 100
    dataset = synthesizer.generate(n)
    dataset.drop(["copy"], axis=1, inplace=True)
    dataset.set_index("subject_id", inplace=True)
    assert len(dataset) == n

    design = rct.generate(n)
    design.set_index("subject_id", inplace=True)
    assert len(design) == n

    dataframe = design.join(dataset, on="subject_id")
    dataframe.to_csv("withdrawal.csv", index=True)


if __name__ == "__main__":
    main()
