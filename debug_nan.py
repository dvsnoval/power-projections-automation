#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from code.aggregator import split_dataframe_by_column

# Test NaN handling
nan_df = pd.DataFrame({
    'category': ['A', np.nan, 'A', 'B', np.nan],
    'value': [1, 2, 3, 4, 5]
})

print("Original DataFrame:")
print(nan_df)
print()

print("Unique values in category column:")
unique_vals = nan_df['category'].unique()
print(unique_vals)
print("Types:", [type(x) for x in unique_vals])
print()

result = split_dataframe_by_column(nan_df, 'category')
print(f"Number of split dataframes: {len(result)}")
print()

for i, df_subset in enumerate(result):
    print(f"Dataframe {i}:")
    print(df_subset)
    if len(df_subset) > 0:
        first_val = df_subset['category'].iloc[0]
        print(f"  First value: {first_val} (type: {type(first_val)})")
        print(f"  Is NaN: {pd.isna(first_val)}")
    print()