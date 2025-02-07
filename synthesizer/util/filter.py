# synthesizer/util/filter.py
"""Util: filter data."""

from typing import List
from dataclasses import dataclass

import pandas as pd


@dataclass
class Filter:
    """Data class to filter data."""

    key: str
    value: str
    comparator: str = "__eq__"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter the DataFrame.

        Args:
            df (pd.DataFrame): the DataFrame

        Returns:
            pd.DataFrame: the filtered DataFrame
        """
        return df.loc[getattr(df[self.key], self.comparator)(self.value)]


@dataclass
class Filters:
    """Data class for applying multiple filters on data."""

    filters: List[Filter]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter the DataFrame in sequence.

        Args:
            df (pd.DataFrame): the DataFrame

        Returns:
            pd.DataFrame: the filtered DataFrame
        """
        for filter in self.filters:
            df = filter(df)
        return df
