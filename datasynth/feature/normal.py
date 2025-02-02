# datasynth/feature/normal.py
"""Feature generation: normal distribution"""

import numpy as np

from .generator import FeatureGenerator


class NormalDistributionGenerator(FeatureGenerator):
    """Generates features from a normal distribution."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """Initializes the generator.

        Args:
            mean (float, optional): mean of the normal distribution. Defaults to 0.0.
            std (float, optional): standard deviation of the normal distribution. Defaults to 1.0.
        """
        self.mean = mean
        self.std = std

    def generate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generates n samples of feature via the given random number generator.

        Args:
            n (int): number of samples to generate
            rng (np.random.Generator): the random number generator to use

        Returns:
            np.ndarray: the generated feature values
        """

        return rng.normal(loc=self.mean, scale=self.std, size=n)
