# synthesizer/feature/exponential.py
"""Feature generation: exponential distribution"""

from typing import Union

import numpy as np

from .generator import FeatureGenerator


class ExponentialDistributionGenerator(FeatureGenerator):
    """Generates features from a exponential distribution."""

    def __init__(
        self,
        scale: float = 1.0,
        ceil: Union[bool, None] = None,
        floor: Union[bool, None] = None,
        decimals: Union[int, None] = None,
    ):
        """Initializes the generator.

        Args:
            scale (float, optional): scale parameter of the exponential distribution. Defaults to 1.0.
            ceil (Union[bool, None], optional): whether to ceil the values. Defaults to None.
            floor (Union[bool, None], optional): whether to floor the values. Defaults to None.
            decimals (Union[int, None], optional): number of decimal places to round to. Defaults to None.
        """
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

        exponential = rng.exponential(scale=self.scale, size=n)
        if self.ceil:
            return np.ceil(exponential)
        elif self.floor:
            return np.floor(exponential)
        elif self.decimals:
            return np.around(exponential, decimals=self.decimals)
        else:
            return exponential
