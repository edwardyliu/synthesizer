# synthesizer/feature/gamma.py
"""Feature generation: gamma distribution"""

from typing import Union

import numpy as np

from .generator import FeatureGenerator


class GammaDistributionGenerator(FeatureGenerator):
    """Generates features from a gamma distribution."""

    def __init__(
        self,
        shape: float = 2.0,
        scale: float = 2.0,
        ceil: Union[bool, None] = None,
        floor: Union[bool, None] = None,
        decimals: Union[int, None] = None,
    ):
        """Initializes the generator.

        Args:
            shape (float, optional): shape parameter of the gamma distribution. Defaults to 2.0.
            scale (float, optional): scale parameter of the gamma distribution. Defaults to 2.0.
            ceil (Union[bool, None], optional): whether to ceil the values. Defaults to None.
            floor (Union[bool, None], optional): whether to floor the values. Defaults to None.
            decimals (Union[int, None], optional): number of decimal places to round to. Defaults to None.
        """
        self.shape = shape
        self.scale = scale

        self.ceil = ceil
        self.floor = floor
        self.decimals = decimals

    def generate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generates n samples of feature via the given random number generator.

        Args:
            n (int): number of samples to generate
            rng (np.random.Generator): the random number generator to use

        Returns:
            np.ndarray: the generated feature values
        """

        gamma = rng.gamma(shape=self.shape, scale=self.scale, size=n)
        if self.ceil:
            return np.ceil(gamma)
        elif self.floor:
            return np.floor(gamma)
        elif self.decimals:
            return np.around(gamma, decimals=self.decimals)
        else:
            return gamma
