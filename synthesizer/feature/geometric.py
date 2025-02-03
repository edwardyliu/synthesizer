# synthesizer/feature/geometric.py
"""Feature generation: geometric distribution"""

from typing import Union

import numpy as np

from .generator import FeatureGenerator


class GeometricDistributionGenerator(FeatureGenerator):
    """Generates features from a geometric distribution."""

    def __init__(
        self,
        success: float = 0.5,
    ):
        """Initializes the generator.

        Args:
            success (float): the probability of success. Defaults to 0.5.
        """
        self.success = success

    def generate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generates n samples of feature via the given random number generator.

        Args:
            n (int): number of samples to generate
            rng (np.random.Generator): the random number generator to use

        Returns:
            np.ndarray: the generated feature values
        """

        return rng.geometric(p=self.success, size=n)
