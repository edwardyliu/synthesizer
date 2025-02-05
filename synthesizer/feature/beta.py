# synthesizer/feature/beta.py
"""Feature generation: beta distribution"""

from typing import Union

import numpy as np

from .generator import FeatureGenerator


class BetaDistributionGenerator(FeatureGenerator):
    """Generates features from a beta distribution."""

    def __init__(
        self,
        a: float = 2,
        b: float = 2,
        ceil: Union[bool, None] = None,
        floor: Union[bool, None] = None,
        decimals: Union[int, None] = None,
        min: Union[float, None] = None,
        max: Union[float, None] = None,
    ):
        """Initializes the generator.

        Args:
            a (float, optional): shape parameter alpha of the beta distribution. Defaults to 2.
            b (float, optional): shape parameter beta of the beta distribution. Defaults to 2.
            ceil (Union[bool, None], optional): whether to ceil the values. Defaults to None.
            floor (Union[bool, None], optional): whether to floor the values. Defaults to None.
            decimals (Union[int, None], optional): number of decimal places to round to. Defaults to None.
            min (Union[float, None], optional): minimum value of the feature. Defaults to None.
            max (Union[float, None], optional): maximum value of the feature. Defaults to None.
        """
        self.a = a
        self.b = b

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

        beta = rng.beta(a=self.a, b=self.b, size=n)
        if self.ceil:
            beta = np.ceil(beta)
        elif self.floor:
            beta = np.floor(beta)
        elif self.decimals:
            beta = np.around(beta, decimals=self.decimals)

        if self.min is not None:
            beta = np.maximum(beta, self.min)
        if self.max is not None:
            beta = np.minimum(beta, self.max)

        return beta
