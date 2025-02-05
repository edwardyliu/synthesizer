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
        min: Union[float, None] = None,
        max: Union[float, None] = None,
    ):
        """Initializes the generator.

        Args:
            scale (float, optional): scale parameter of the exponential distribution. Defaults to 1.0.
            ceil (Union[bool, None], optional): whether to ceil the values. Defaults to None.
            floor (Union[bool, None], optional): whether to floor the values. Defaults to None.
            decimals (Union[int, None], optional): number of decimal places to round to. Defaults to None.
            min (Union[float, None], optional): minimum value of the feature. Defaults to None.
            max (Union[float, None], optional): maximum value of the feature. Defaults to None.
        """
        self.scale = scale

        self.ceil = ceil
        self.floor = floor
        self.decimals = decimals
        self.min = min
        self.max = max

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
            exponential = np.ceil(exponential)
        elif self.floor:
            exponential = np.floor(exponential)
        elif self.decimals:
            exponential = np.around(exponential, decimals=self.decimals)

        if self.min is not None:
            exponential = np.maximum(exponential, self.min)
        if self.max is not None:
            exponential = np.minimum(exponential, self.max)

        return exponential
