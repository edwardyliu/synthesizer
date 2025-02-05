# synthesizer/feature/f.py
"""Feature generation: f distribution"""

from typing import Union

import numpy as np

from .generator import FeatureGenerator


class FDistributionGenerator(FeatureGenerator):
    """Generates features from a f distribution."""

    def __init__(
        self,
        dfnum: float = 1.0,
        dfden: float = 48.0,
        ceil: Union[bool, None] = None,
        floor: Union[bool, None] = None,
        decimals: Union[int, None] = None,
        min: Union[float, None] = None,
        max: Union[float, None] = None,
    ):
        """Initializes the generator.

        Args:
            dfnum (float, optional): numerator degrees of freedom. Defaults to 1.0.
            dfden (float, optional): denominator degrees of freedom. Defaults to 48.0.
            ceil (Union[bool, None], optional): whether to ceil the values. Defaults to None.
            floor (Union[bool, None], optional): whether to floor the values. Defaults to None.
            decimals (Union[int, None], optional): number of decimal places to round to. Defaults to None.
            min (Union[float, None], optional): minimum value of the feature. Defaults to None.
            max (Union[float, None], optional): maximum value of the feature. Defaults to None.
        """
        self.dfnum = dfnum
        self.dfden = dfden

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

        f = rng.f(dfnum=self.dfnum, dfden=self.dfden, size=n)
        if self.ceil:
            f = np.ceil(f)
        elif self.floor:
            f = np.floor(f)
        elif self.decimals:
            f = np.around(f, decimals=self.decimals)

        if self.min is not None:
            f = np.maximum(f, self.min)
        if self.max is not None:
            f = np.minimum(f, self.max)

        return f
