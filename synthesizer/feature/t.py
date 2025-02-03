# synthesizer/feature/t.py
"""Feature generation: Student's t distribution"""

from typing import Union

import numpy as np

from .generator import FeatureGenerator


class TDistributionGenerator(FeatureGenerator):
    """Generates features from a Student's t distribution."""

    def __init__(
        self,
        df: float = 1.0,
        ceil: Union[bool, None] = None,
        floor: Union[bool, None] = None,
        decimals: Union[int, None] = None,
    ):
        """Initializes the generator.

        Args:
            df (float, optional): degrees of freedom. Defaults to 1.0.
            ceil (Union[bool, None], optional): whether to ceil the values. Defaults to None.
            floor (Union[bool, None], optional): whether to floor the values. Defaults to None.
            decimals (Union[int, None], optional): number of decimal places to round to. Defaults to None.
        """
        self.df = df

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

        t = rng.standard_t(df=self.df, size=n)
        if self.ceil:
            return np.ceil(t)
        elif self.floor:
            return np.floor(t)
        elif self.decimals:
            return np.around(t, decimals=self.decimals)
        else:
            return t
