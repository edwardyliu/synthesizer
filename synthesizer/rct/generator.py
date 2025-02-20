# synthesizer/rct/generator.py
"""Base class for RCT generation."""

from typing import Dict
from abc import ABC, abstractmethod

import pandas as pd

from synthesizer.util import Filters


class RCTGenerator(ABC):
    """Abstract class for RCT generation."""

    @abstractmethod
    def generate(
        self,
        n: int,
        sid: str = "subject_id",
        sid_start: int = 0,
    ) -> pd.DataFrame:
        """Generate a DataFrame consisting columns that define the RCT design.

        Args:
            n (int): number of subjects
            sid (str, optional): name of the subject id column. Defaults to "subject_id".
            sid_start (int, optional): id number to start with, exclusive

        Returns:
            pd.DataFrame: the generated RCT design DataFrame
        """


class RCTGeneratorGrouped(RCTGenerator):
    """Abstract class for grouped RCT generation."""

    @abstractmethod
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
