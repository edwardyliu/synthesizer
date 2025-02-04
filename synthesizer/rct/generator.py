# synthesizer/rct/generator.py
"""Base class for RCT generation."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class RCTGenerator(ABC):
    """Abstract class for RCT generation."""

    @abstractmethod
    def generate(
        self,
        n: int,
        rng: np.random.Generator,
        sid: str = "subject_id",
        sid_start: int = 0,
    ) -> pd.DataFrame:
        """Generate a DataFrame consisting columns that define the RCT design.

        Args:
            n (int): number of subjects
            rng (np.random.Generator): the random number generator to use
            sid (str, optional): name of the subject id column. Defaults to "subject_id".
            sid_start (int, optional): id number to start with, exclusive

        Returns:
            pd.DataFrame: the generated RCT design DataFrame
        """
