# synthesizer/blocking/generator.py
"""Base class for Blocking generation."""

from typing import Dict
from abc import ABC, abstractmethod

import pandas as pd

from synthesizer.util import Filters


class BlockingGenerator(ABC):
    """Abstract class for Blocking generation."""

    @abstractmethod
    def generate(
        self,
        subjects: pd.DataFrame,
        blockings: Dict[str, Filters],
    ) -> pd.DataFrame:
        """Generate a DataFrame consisting columns that define the blocking design.

        Args:
            subjects (pd.DataFrame): DataFrame of subjects
            blockings (Dict[str, Filters]): list of blockings and their associated filters

        Returns:
            pd.DataFrame: the generated blocking designed DataFrame
        """


class SimpleBlockingGenerator(BlockingGenerator):
    """Simple blocking generator."""

    def generate(
        self,
        subjects: pd.DataFrame,
        blockings: Dict[str, Filters],
    ) -> pd.DataFrame:
        """Generate a DataFrame consisting columns that define the blocking design.

        Args:
            subjects (pd.DataFrame): DataFrame of subjects
            blockings (Dict[str, Filters]): list of blockings and their associated filters

        Returns:
            pd.DataFrame: the generated blocking designed DataFrame
        """

        # blocking subjects
        blocks = {}
        for key, filters in blockings.items():
            blocks[key] = filters(subjects)

        data = {}
        for key, block in blocks.items():
            n = len(block)
            data["block"] = data.get("block", []) + [key] * n

            # for each column in DataFrame block, populate to data
            for col in block.columns:
                data[col] = data.get(col, []) + block[col].tolist()

        return pd.DataFrame(data)
