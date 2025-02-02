# datasynth/feature/normal.py
"""Feature generation: normal distribution"""

from typing import Union

import numpy as np

from .generator import FeatureGenerator


class NormalDistributionGenerator(FeatureGenerator):
    """Generates features from a normal distribution."""

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        round: Union[int, None] = None,
        ceil: Union[bool, None] = None,
        floor: Union[bool, None] = None,
    ):
        """Initializes the generator.

        Args:
            mean (float, optional): mean of the normal distribution. Defaults to 0.0.
            std (float, optional): standard deviation of the normal distribution. Defaults to 1.0.
        """
        self.mean = mean
        self.std = std
        self.round = round
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

        normal = rng.normal(loc=self.mean, scale=self.std, size=n)
        if self.ceil:
            return np.ceil(normal)
        elif self.floor:
            return np.floor(normal)
        elif self.round:
            return np.around(normal, decimals=self.round)
        else:
            return normal
