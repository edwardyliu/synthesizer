# synthesizer/rct/withdrawal.py
"""RCT generation: withdrawal RCT design"""

import numpy as np
import pandas as pd

from .generator import RCTGenerator


class WithdrawalRCTGenerator(RCTGenerator):
    """Class for withdrawal RCT generation."""

    def __init__(
        self,
        seed: int = None,
        response: float = 0.3,
        withdrawal: float = 0.5,
        **kwargs,
    ):
        """Initializes the generator."""

        self.rng = np.random.default_rng(seed)
        self.response = response
        self.withdrawal = withdrawal

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
        sid: str = "subject_id",
    ) -> pd.DataFrame:
        """Generate a DataFrame consisting columns that define the RCT design.

        Args:
            subjects (pd.DataFrame): DataFrame of subjects
            sid (str, optional): name of the subject id column. Defaults to "subject_id".

        Returns:
            pd.DataFrame: the generated RCT design DataFrame
        """

        data = {}

        # simulate response
        responses = self.rng.binomial(n=1, p=self.response, size=len(subjects))
        data["response"] = [True if response == 1 else False for response in responses]

        # simulate withdrawal status
        # i.e. randomize responders to continue or to withdraw
        # continue: True
        # withdraw: False
        # not responder: None
        statuses = []
        for response in responses:
            if response:
                statuses.append(
                    False if self.rng.binomial(n=1, p=self.withdrawal) == 1 else True
                )
            else:
                statuses.append(None)
        data["status"] = statuses

        # randomly assign subjects to an arm
        for idx, status in enumerate(statuses):
            data[sid] = data.get(sid, []) + [subjects.iloc[idx][sid]]
            if status:
                arm_assignment = self.rng.integers(low=0, high=len(self.arms))
                for key, value in self.arms[arm_assignment].items():
                    data[key] = data.get(key, []) + [value]
            else:
                for key in self.arms[0].keys():
                    data[key] = data.get(key, []) + [None]

            # for each column in DataFrame subjects, populate to data
            # except sid
            for col in subjects.columns:
                if col != sid:
                    data[col] = data.get(col, []) + [subjects.iloc[idx][col]]

        return pd.DataFrame(data)
