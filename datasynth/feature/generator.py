# datasynth/feature/generator.py
"""Base class for feature generation."""

from abc import ABC, abstractmethod

import numpy as np


class FeatureGenerator(ABC):
    """Abstract class for feature generation."""

    @abstractmethod
    def generate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generates n samples of feature via the given random number generator.

        Args:
            n (int): number of samples to generate
            rng (np.random.Generator): the random number generator to use

        Returns:
            np.ndarray: the generated feature values
        """
