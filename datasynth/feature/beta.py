# datasynth/feature/beta.py
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
        round: Union[int, None] = None,
    ):
        """Initializes the generator.

        Args:
            a (float, optional): shape parameter alpha of the beta distribution. Defaults to 2.
            b (float, optional): shape parameter beta of the beta distribution. Defaults to 2.
            ceil (Union[bool, None], optional): whether to ceil the values. Defaults to None.
            floor (Union[bool, None], optional): whether to floor the values. Defaults to None.
            round (Union[int, None], optional): number of decimal places to round to. Defaults to None.
        """
        self.a = a
        self.b = b

        self.ceil = ceil
        self.floor = floor
        self.round = round

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
            return np.ceil(beta)
        elif self.floor:
            return np.floor(beta)
        elif self.round:
            return np.around(beta, decimals=self.round)
        else:
            return beta
