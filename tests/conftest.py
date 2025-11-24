"""Shared test fixtures and configuration."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import from code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_dataframe():
    """Standard test DataFrame fixture."""
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "z": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )


@pytest.fixture
def categorical_dataframe():
    """DataFrame with categorical data for splitting tests."""
    return pd.DataFrame(
        {
            "category": ["A", "B", "A", "C", "B", "A", "C"],
            "value": [1, 2, 3, 4, 5, 6, 7],
            "score": [10.5, 20.1, 30.2, 40.3, 50.4, 60.5, 70.6],
        }
    )


@pytest.fixture
def dataframe_with_nan():
    """DataFrame containing NaN values for testing edge cases."""
    return pd.DataFrame(
        {"category": ["A", np.nan, "A", "B", np.nan], "value": [1, 2, 3, 4, 5]}
    )
