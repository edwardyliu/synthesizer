# datasynth/synthesizer.py

from typing import Dict, List, Union

import numpy as np
import pandas as pd

from datasynth.feature import FeatureGenerator


class DataSynthesizer:
    """Class for dataset generation"""

    def __init__(
        self,
        seed: int = None,
        copies: int = 1,
        static: List[str] = [],
    ):
        """Initializes the synthesizer with a random seed.

        Args:
            seed (int, optional): random seed for reproducibility. Defaults to None.
            copies (Union[int, None], optional): number of copies per subject. Defaults to 1.
            static (Union[List[str], str], optional): columns to keep static per subject. Defaults to [].
        """

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.generators: Dict[str, FeatureGenerator] = {}
        self.duplicates: Dict[str, str] = {}
        self.copies = copies
        self.static: List[str] = static

    def add_feature(
        self,
        name: str,
        generator: Union[FeatureGenerator, None] = None,
        duplicates: Union[List[str], str, None] = None,
    ):
        """Adds a feature generator to the synthesizer.

        Args:
            name (str): name of the feature
            generator (FeatureGenerator): the feature generator. Defaults to None.
            duplicates (Union[List[str], str], optional): columns to duplicate this feature. Defaults to None.
        """

        if generator:
            self.generators[name] = generator

        if duplicates:
            if isinstance(duplicates, list):
                for duplicate in duplicates:
                    self.duplicates[duplicate] = name
            elif isinstance(duplicates, str):
                self.duplicates[duplicates] = name

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

        dataset = {}
        # +sid
        for copy in range(1, self.copies + 1, 1):
            dataset[sid] = dataset.get(sid, []) + list(range(1, n + 1, 1))
            dataset["copy"] = dataset.get("copy", []) + [copy] * n

        # +generator
        for name, generator in self.generators.items():
            if name in self.static:
                dataset[name] = np.tile(generator.generate(n, self.rng), self.copies)
            else:
                for _ in range(self.copies):
                    if isinstance(dataset.get(name), np.ndarray):
                        dataset[name] = np.concatenate(
                            [dataset[name], generator.generate(n, self.rng)], axis=0
                        )
                    else:
                        dataset[name] = generator.generate(n, self.rng)

        # +duplicates
        for duplicate, value in self.duplicates.items():
            dataset[duplicate] = dataset[value]

        return pd.DataFrame(dataset)
