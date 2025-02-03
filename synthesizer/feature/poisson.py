# synthesizer/feature/poisson.py
"""Feature generation: poisson distribution"""

import numpy as np

from .generator import FeatureGenerator


class PoissonDistributionGenerator(FeatureGenerator):
    """Generates features from a poisson distribution."""

    def __init__(
        self,
        events: int = 10,
    ):
        """Initializes the generator.

        Args:
            events (int): expected number of events occurring in a fixed-time interval, must be >= 0. Defaults to 10.
        """
        self.events = events

    def generate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generates n samples of feature via the given random number generator.

        Args:
            n (int): number of samples to generate
            rng (np.random.Generator): the random number generator to use

        Returns:
            np.ndarray: the generated feature values
        """

        return rng.poisson(lam=self.events, size=n)
