# Test: datasynth/feature/choice.py

import pytest

import numpy as np

from datasynth.feature import ChoiceDistributionGenerator


def test_choice():
    """Test Unit: datasynth.feature.choice"""

    categories = ["A", "B", "C"]
    probabilities = [0.4, 0.3, 0.3]
    n = 1000
    rng = np.random.default_rng(28)

    generator = ChoiceDistributionGenerator(categories, probabilities)
    feature = generator.generate(n, rng)

    assert (feature == "A").sum() / n == pytest.approx(0.4, abs=0.1)
    assert (feature == "B").sum() / n == pytest.approx(0.3, abs=0.1)
    assert (feature == "C").sum() / n == pytest.approx(0.3, abs=0.1)
