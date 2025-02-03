# synthesizer/rct/crossover.py
"""RCT generation: cross-over RCT design"""

import itertools

import numpy as np
import pandas as pd

from .generator import RCTGenerator


class CrossoverRCTGenerator(RCTGenerator):
    """Class for cross-over RCT generation."""

    def __init__(self, seed: int = None, ncopies=2, **kwargs):
        """Initializes the generator."""

        self.rng = np.random.default_rng(seed)
        self.ncopies = ncopies

        # generate all possible arms
        keys, values = zip(*kwargs.items())
        self.arms = [dict(zip(keys, v)) for v in itertools.product(*values)]
        self.arm_combinations = list(
            itertools.combinations(
                self.arms,
                min(
                    self.ncopies,
                    len(self.arms),
                ),
            ),
        )

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

        # randomly assign subjects to an arm permutation
        arm_permutation_assignments = self.rng.integers(
            low=0,
            high=len(self.arm_combinations),
            size=n,
        )

        # create the RCT design DataFrame
        data = {}
        for idx, assignment in enumerate(arm_permutation_assignments, 1):
            for copy, arm in enumerate(self.arm_combinations[assignment], 1):
                data[sid] = data.get(sid, []) + [idx]
                data["copy"] = data.get("copy", []) + [copy]
                for key, value in arm.items():
                    data[key] = data.get(key, []) + [value]

        return pd.DataFrame(data)
