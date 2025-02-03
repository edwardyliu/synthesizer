# datasynth/synthesizer.py

from typing import Dict, List, Union

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
        self.generators: Dict[str, FeatureGenerator] = {}
        self.duplicates: Dict[str, str] = {}

    def add_feature(
        self,
        name: str,
        generator: Union[FeatureGenerator, None] = None,
        duplicate: Union[List[str], str, None] = None,
    ):
        """Adds a feature generator to the synthesizer.

        Args:
            name (str): name of the feature
            generator (FeatureGenerator): the feature generator. Defaults to None.
            duplicate (Union[List[str], str], optional): names of features duplicating this feature. Defaults to None.
        """

        if generator:
            self.generators[name] = generator

        if duplicate:
            if isinstance(duplicate, list):
                for dup in duplicate:
                    self.duplicates[dup] = name
            elif isinstance(duplicate, str):
                self.duplicates[duplicate] = name

    def generate(
        self,
        n: int,
        sid: str = "subject_id",
    ) -> pd.DataFrame:
        """Generates a dataset of pd.DataFrame of size n.

        Args:
            n (int): size of the dataset
            sid (str, optional): name of the subject id column. Defaults to "subject_id".

        Returns:
            pd.DataFrame: the generated dataset
        """

        dataset = {sid: np.arange(start=1, stop=n + 1)}
        for name, generator in self.generators.items():
            dataset[name] = generator.generate(n, self.rng)

        for dup, name in self.duplicates.items():
            dataset[dup] = dataset[name]

        return pd.DataFrame(dataset)
