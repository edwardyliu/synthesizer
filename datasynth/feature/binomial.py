# datasynth/feature/binomial.py
"""Feature generation: binomial distribution"""

import numpy as np

from .generator import FeatureGenerator


class BinomialDistributionGenerator(FeatureGenerator):
    """Generates features from a binomial distribution."""

    def __init__(
        self,
        trials: int = 10,
        success: float = 0.5,
    ):
        """Initializes the generator.

        Args:
            trials (int): number of trials. Defaults to 10.
            success (float): probability of success in each trial. Defaults to 0.5.
        """
        self.trials = trials
        self.success = success

    def generate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generates n samples of feature via the given random number generator.

        Args:
            n (int): number of samples to generate
            rng (np.random.Generator): the random number generator to use

        Returns:
            np.ndarray: the generated feature values
        """

        return rng.binomial(n=self.trials, p=self.success, size=n)
