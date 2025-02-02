# datasynth/feature/binomial.py
"""Feature generation: binomial distribution"""

from typing import Union

import numpy as np

from .generator import FeatureGenerator


class BinomialDistributionGenerator(FeatureGenerator):
    """Generates features from a binomial distribution."""

    def __init__(
        self,
        trials: int = 10,
        success: float = 0.5,
        ceil: Union[bool, None] = None,
        floor: Union[bool, None] = None,
    ):
        """Initializes the generator.

        Args:
            trials (int): number of trials. Defaults to 10.
            success (float): probability of success in each trial. Defaults to 0.5.
            ceil (Union[bool, None], optional): whether to ceil the values. Defaults to None.
            floor (Union[bool, None], optional): whether to floor the values. Defaults to None.
        """
        self.trials = trials
        self.success = success

        self.ceil = ceil
        self.floor = floor

    def generate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generates n samples of feature via the given random number generator.

        Args:
            n (int): number of samples to generate
            rng (np.random.Generator): the random number generator to use

        Returns:
            np.ndarray: the generated feature values
        """

        binomial = rng.binomial(n=self.trials, p=self.success, size=n)
        if self.ceil:
            return np.ceil(binomial)
        elif self.floor:
            return np.floor(binomial)
        else:
            return binomial
