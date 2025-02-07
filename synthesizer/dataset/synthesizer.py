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
        self.generators: Dict[str, List[FeatureGenerator]] = {}
        self.duplicates: Dict[str, str] = {}
        self.static: List[str] = static

    def add_feature(
        self,
        name: str,
        generators: Union[List[FeatureGenerator], FeatureGenerator],
        duplicates: Union[List[str], str, None] = None,
    ):
        """Adds a feature generator to the synthesizer.

        Args:
            name (str): name of the feature
            generators (Union[List[FeatureGenerator], FeatureGenerator): the feature generators.
            duplicates (Union[List[str], str], optional): columns to duplicate this feature. Defaults to None.
        """

        if isinstance(generators, FeatureGenerator):
            generators = [generators]
        self.generators[name] = generators

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

        def _generate(size: int, generators: List[FeatureGenerator]) -> np.ndarray:
            values = []
            for _ in range(size):
                choice = self.rng.integers(low=0, high=len(generators))
                generator = generators[choice]
                value = generator.generate(1, self.rng)
                values.append(value)
            return np.concatenate(values, axis=0)

        # +generator
        for name, generators in self.generators.items():
            if name in self.static:
                subset = _generate(n, generators)
                dataset[name] = np.tile(subset, ncopies)
            else:
                for _ in range(ncopies):
                    subset = _generate(n, generators)
                    if isinstance(dataset.get(name), np.ndarray):
                        dataset[name] = np.concatenate([dataset[name], subset], axis=0)
                    else:
                        dataset[name] = subset

        # +duplicates
        for duplicate, value in self.duplicates.items():
            dataset[duplicate] = dataset[value]

        return pd.DataFrame(dataset)
