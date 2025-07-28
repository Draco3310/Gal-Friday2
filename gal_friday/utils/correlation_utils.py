"""Utilities for computing correlation matrices.

This module provides functions for calculating correlation
coefficients between features in dataframes.
"""

import pandas as pd


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr()
