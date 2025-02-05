# synthesizer/dataset/synthesizer.py

from typing import Dict, List, Union

import numpy as np
import pandas as pd

from synthesizer.feature import FeatureGenerator


class DatasetSynthesizer:
    """Class for dataset generation"""

    def __init__(
        self,
        seed: int = None,
        static: List[str] = [],
    ):
        """Initializes the synthesizer with a random seed.

        Args:
            seed (int, optional): random seed for reproducibility. Defaults to None.
            static (Union[List[str], str], optional): columns to keep static per subject. Defaults to [].
        """

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.generators: Dict[str, FeatureGenerator] = {}
        self.duplicates: Dict[str, str] = {}
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
        sid_start: int = 0,
        ncopies: int = 1,
    ) -> pd.DataFrame:
        """Generates a dataset of pd.DataFrame of size n.

        Args:
            n (int): size of the dataset
            sid (str, optional): name of the subject id column. Defaults to "subject_id".
            sid_start (int, optional): id number to start with, exclusive
            ncopies (int, optional): number of copies per subject. Defaults to 1.

        Returns:
            pd.DataFrame: the generated dataset
        """

        dataset = {}
        # +sid
        for copy in range(1, ncopies + 1, 1):
            dataset[sid] = dataset.get(sid, []) + list(
                range(sid_start + 1, sid_start + 1 + n, 1)
            )
            dataset["copy"] = dataset.get("copy", []) + [copy] * n

        # +generator
        for name, generator in self.generators.items():
            if name in self.static:
                dataset[name] = np.tile(generator.generate(n, self.rng), ncopies)
            else:
                for _ in range(ncopies):
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
