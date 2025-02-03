# datasynth/rct/factorial.py
"""RCT generation: factorial RCT design"""

import itertools

import numpy as np
import pandas as pd

from .generator import RCTGenerator


class FactorialRCTGenerator(RCTGenerator):
    """Class for factorial RCT generation."""

    def __init__(self, seed: int = None, **kwargs):
        """Initializes the generator."""

        self.rng = np.random.default_rng(seed)

        # generate all possible arms
        keys, values = zip(*kwargs.items())
        self.arms = [dict(zip(keys, v)) for v in itertools.product(*values)]

    def generate(
        self,
        n: int,
        sid: str = "subject_id",
    ) -> pd.DataFrame:
        """Generate a DataFrame consisting columns that define the RCT design.

        Args:
            n (int): number of subjects
            rng (np.random.Generator): the random number generator to use
            sid (str, optional): name of the subject id column. Defaults to "subject_id".

        Returns:
            pd.DataFrame: the generated RCT design DataFrame
        """

        # randomly assign subjects to an arm
        arm_assignments = self.rng.integers(low=0, high=len(self.arms), size=n)

        # create the RCT design DataFrame
        data = {}
        for idx, assignment in enumerate(arm_assignments, 1):
            data[sid] = data.get(sid, []) + [idx]
            for key, value in self.arms[assignment].items():
                data[key] = data.get(key, []) + [value]

        return pd.DataFrame(data)
