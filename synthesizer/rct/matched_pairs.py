# synthesizer/rct/matched_pairs.py
"""RCT generation: matched_pairs RCT design"""

from typing import Dict

import numpy as np
import pandas as pd

from .generator import RCTGeneratorGrouped

from synthesizer.util import Filters


class MatchedPairsRCTGenerator(RCTGeneratorGrouped):
    """Class for matched_pairs RCT generation."""

    def __init__(self, seed: int = None, **kwargs):
        """Initializes the generator."""

        self.rng = np.random.default_rng(seed)

        # generate all possible arms
        placebo = False
        arms = []
        for treatment, levels in kwargs.items():
            for level in levels:
                arm = {}
                for name in kwargs.keys():
                    arm[name] = "placebo" if treatment != name else level

                if all(v == "placebo" for v in arm.values()):
                    if not placebo:
                        placebo = True
                    else:
                        continue

                arms.append(arm)

        self.arms = arms

    def generate(
        self,
        subjects: pd.DataFrame,
        groupings: Dict[str, Filters],
        sid: str = "subject_id",
    ) -> pd.DataFrame:
        """Generate a DataFrame consisting columns that define the RCT design.

        Args:
            subjects (pd.DataFrame): DataFrame of subjects
            groupings (Dict[str, Filters]): list of groupings and their associated filters
            sid (str, optional): name of the subject id column. Defaults to "subject_id".

        Returns:
            pd.DataFrame: the generated RCT design DataFrame
        """

        # subgroup subjects
        subgroups = {}
        for key, filters in groupings.items():
            subgroups[key] = filters(subjects)

        data = {}
        for key, subgroup in subgroups.items():
            n = len(subgroup)
            data["group"] = data.get("group", []) + [key] * n

            # randomly assign subjects of subgroup to an arm
            arm_assignments = self.rng.integers(low=0, high=len(self.arms), size=n)

            # create the RCT design for subjects of subgroup
            for idx, assignment in enumerate(arm_assignments):
                data[sid] = data.get(sid, []) + [subgroup.iloc[idx][sid]]
                for key, value in self.arms[assignment].items():
                    data[key] = data.get(key, []) + [value]

                # for each column in DataFrame subjects, populate to data
                # except sid
                for col in subgroup.columns:
                    if col != sid:
                        data[col] = data.get(col, []) + [subgroup.iloc[idx][col]]

        return pd.DataFrame(data)
