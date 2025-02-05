# synthesizer/feature/logistic.py
"""Feature generation: logistic distribution"""

from typing import Union

import numpy as np

from .generator import FeatureGenerator


class LogisticDistributionGenerator(FeatureGenerator):
    """Generates features from a logistic distribution."""

    def __init__(
        self,
        mean: float = 0.0,
        scale: float = 1.0,
        ceil: Union[bool, None] = None,
        floor: Union[bool, None] = None,
        decimals: Union[int, None] = None,
        min: Union[float, None] = None,
        max: Union[float, None] = None,
    ):
        """Initializes the generator.

        Args:
            mean (float, optional): mean of the logistic distribution. Defaults to 0.0.
            std (float, optional): standard deviation of the logistic distribution. Defaults to 1.0.
            ceil (Union[bool, None], optional): whether to ceil the values. Defaults to None.
            floor (Union[bool, None], optional): whether to floor the values. Defaults to None.
            decimals (Union[int, None], optional): number of decimal places to round to. Defaults to None.
            min (Union[float, None], optional): minimum value of the feature. Defaults to None.
            max (Union[float, None], optional): maximum value of the feature. Defaults to None.
        """
        self.mean = mean
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

        logistic = rng.logistic(loc=self.mean, scale=self.scale, size=n)
        if self.ceil:
            logistic = np.ceil(logistic)
        elif self.floor:
            logistic = np.floor(logistic)
        elif self.decimals:
            logistic = np.around(logistic, decimals=self.decimals)

        if self.min is not None:
            logistic = np.maximum(logistic, self.min)
        if self.max is not None:
            logistic = np.minimum(logistic, self.max)

        return logistic
