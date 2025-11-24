"""Aggregation and interpolation utilities for pandas DataFrames."""

from .aggregator import interpolate_dataframe, split_dataframe_by_column

__version__ = "0.1.0"
__all__ = ["interpolate_dataframe", "split_dataframe_by_column"]
