# datasynth/synthesizer.py

from typing import Dict

import numpy as np
import pandas as pd

from datasynth.feature import FeatureGenerator


class DataSynthesizer:
    """Class for dataset generation"""

    def __init__(self, seed: int = None):
        """Initializes the synthesizer with a random seed.

        Args:
            seed (int, optional): random seed for reproducibility. Defaults to None.
        """

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.feature_generators: Dict[str, FeatureGenerator] = {}

    def add_feature(self, name: str, generator: FeatureGenerator):
        """Adds a feature generator to the synthesizer.

        Args:
            name (str): name of the feature
            generator (FeatureGenerator): the feature generator
        """

        self.feature_generators[name] = generator

    def generate(
        self,
        n: int,
        sid: str = "subject_id",
    ) -> pd.DataFrame:
        """Generates a dataset of pd.DataFrame of size n.

        Args:
            n (int): size of the dataset

        Returns:
            pd.DataFrame: the generated dataset
        """

        dataset = {sid: np.arange(start=1, stop=n + 1)}
        for name, generator in self.feature_generators.items():
            dataset[name] = generator.generate(n, self.rng)

        return pd.DataFrame(dataset)
