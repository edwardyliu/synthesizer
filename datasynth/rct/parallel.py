# datasynth/rct/parallel.py
"""RCT generation: parallel RCT design"""

import itertools

import numpy as np
import pandas as pd

from .generator import RCTGenerator


class ParallelGenerator(RCTGenerator):
    """Abstract class for RCT generation."""

    def __init__(self, **kwargs):
        """Initializes the generator."""

        keys, values = zip(*kwargs.items())
        self.arms = [dict(zip(keys, v)) for v in itertools.product(*values)]

    def generate(
        self,
        n: int,
        rng: np.random.Generator,
        sid: str = "subject_id",
    ) -> pd.DataFrame:
        """Generate a DataFrame consisting columns that define the RCT design.

        Args:
            n (int): number of subjects
            rng (np.random.Generator): the random number generator to use

        Returns:
            pd.DataFrame: the generated RCT design DataFrame
        """

        # randomly assign subjects to an arm
        arm_assignments = rng.integers(low=0, high=len(self.arms), size=n)

        # create the RCT design DataFrame
        data = {}
        for idx, assignment in enumerate(arm_assignments, 1):
            data[sid] = data.get(sid, []) + [idx]
            for key, value in self.arms[assignment].items():
                data[key] = data.get(key, []) + [value]

        return pd.DataFrame(data)
