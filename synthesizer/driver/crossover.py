# synthesizer/driver/crossover.py

import logging

logger = logging.getLogger("synthesizer")


from synthesizer import DatasetSynthesizer
from synthesizer.feature import (
    ChoiceDistributionGenerator,
    NormalDistributionGenerator,
    UniformIntegerDistributionGenerator,
)
from synthesizer.rct import CrossoverRCTGenerator


def main():
    logger.info("synthesizer.driver.crossover.main: generating dataset")

    # build data synthesizer
    copies = 2
    static = ["age", "occupation"]
    synthesizer = DatasetSynthesizer(copies=copies, static=static)
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
        "income",
        NormalDistributionGenerator(65000, 1.0, decimals=2),
    )

    # design and generate the RCT
    treatments = {
        "discount": ["placebo", "20"],
        "time": ["placebo", "40", "20"],
    }
    rct = CrossoverRCTGenerator(**treatments)

    # generate dataset and design for <n> subjects
    n = 100
    dataset = synthesizer.generate(n)
    dataset.set_index(["subject_id", "copy"], inplace=True)
    assert len(dataset) == copies * n

    design = rct.generate(n)
    design.set_index(["subject_id", "copy"], inplace=True)
    assert len(design) == copies * n

    dataframe = design.join(dataset, on=["subject_id", "copy"])
    dataframe.to_csv("crossover.csv", index=True)


if __name__ == "__main__":
    main()
