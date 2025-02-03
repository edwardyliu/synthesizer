# synthesizer/feature/choice.py
"""Feature generation: choice distribution"""

from typing import List, Union

import numpy as np

from .generator import FeatureGenerator


class ChoiceDistributionGenerator(FeatureGenerator):
    """Generates features from a choice distribution."""

    def __init__(
        self,
        categories: List[str],
        probabilities: Union[List[float], None] = None,
    ):
        """Initializes the generator.

        Args:
            categories (List[str]): list of categories to choose from; e.g. ["A", "B", "C"]
            probabilities (Union[List[float], None], optional):
                list of probabilities for each category summing to 1.
                Must be the same length as <categories>.
                Defaults to None.
        """
        self.categories = categories
        self.probabilities = probabilities

    def generate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generates n samples of feature via the given random number generator.

        Args:
            n (int): number of samples to generate
            rng (np.random.Generator): the random number generator to use

        Returns:
            np.ndarray: the generated feature values
        """

        return rng.choice(self.categories, p=self.probabilities, size=n)
