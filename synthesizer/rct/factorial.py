# synthesizer/rct/factorial.py
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
        subjects: pd.DataFrame,
        sid: str = "subject_id",
    ) -> pd.DataFrame:
        """Generate a DataFrame consisting columns that define the RCT design.

        Args:
            subjects (pd.DataFrame): DataFrame of subjects
            sid (str, optional): name of the subject id column. Defaults to "subject_id".

        Returns:
            pd.DataFrame: the generated RCT design DataFrame
        """

        # randomly assign subjects to an arm
        arm_assignments = self.rng.integers(
            low=0, high=len(self.arms), size=len(subjects)
        )

        # create the RCT design DataFrame
        data = {}
        for idx, assignment in enumerate(arm_assignments):
            data[sid] = data.get(sid, []) + [subjects.iloc[idx][sid]]
            for key, value in self.arms[assignment].items():
                data[key] = data.get(key, []) + [value]

            # for each column in DataFrame subjects, populate to data
            # except sid
            for col in subjects.columns:
                if col != sid:
                    data[col] = data.get(col, []) + [subjects.iloc[idx][col]]

        return pd.DataFrame(data)
