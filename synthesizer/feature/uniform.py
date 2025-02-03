# synthesizer/feature/uniform.py
"""Feature generation: uniform distribution"""

import numpy as np

from .generator import FeatureGenerator


class UniformDistributionGenerator(FeatureGenerator):
    """Generates features from a uniform distribution."""

    def __init__(self, low: float = 0.0, high: float = 1.0):
        """Initializes the generator.

        Args:
            low (float, optional): low of the uniform distribution. Defaults to 0.0.
            high (float, optional): high of the uniform distribution. Defaults to 1.0.
        """
        self.low = low
        self.high = high

    def generate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generates n samples of feature via the given random number generator.

        Args:
            n (int): number of samples to generate
            rng (np.random.Generator): the random number generator to use

        Returns:
            np.ndarray: the generated feature values
        """

        return rng.uniform(low=self.low, high=self.high, size=n)


class UniformIntegerDistributionGenerator(FeatureGenerator):
    """Generates features from a uniform distribution as integers."""

    def __init__(self, low: int = 0, high: int = 1):
        """Initializes the generator.

        Args:
            low (int, optional): low of the uniform distribution. Defaults to 0.
            high (int, optional): high of the uniform distribution. Defaults to 1.
        """
        self.low = low
        self.high = high

    def generate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generates n samples of feature via the given random number generator.

        Args:
            n (int): number of samples to generate
            rng (np.random.Generator): the random number generator to use

        Returns:
            np.ndarray: the generated feature values
        """

        return rng.integers(low=self.low, high=self.high, size=n)
