import pandas as pd
import numpy as np


def split_dataframe_by_column(df, column_name):
    """
    Splits a DataFrame into a dictionary of DataFrames, each with constant values in the specified column.

    Parameters:
    df (pd.DataFrame): Input DataFrame to split
    column_name (str): Name of the column to use for splitting

    Returns:
    dict: Dictionary where keys are unique values from column_name and values are DataFrames
          For NaN values, the key will be the string 'NaN'
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    # Get unique values in the specified column
    unique_values = df[column_name].unique()

    # Create dictionary of dataframes, one for each unique value
    dataframes = {}
    for value in unique_values:
        if pd.isna(value):
            # Handle NaN values separately since NaN != NaN
            subset_df = df[df[column_name].isna()].copy()
            key = "NaN"
        else:
            subset_df = df[df[column_name] == value].copy()
            key = value

        if len(subset_df) > 0:  # Only add non-empty dataframes
            dataframes[key] = subset_df

    return dataframes


def interpolate_dataframe(df, executive_metric, step):
    """
    Interpolates a DataFrame based on an executive metric column.

    Parameters:
    df (pd.DataFrame): Input DataFrame with numeric columns
    executive_metric (str): Name of the column to use as the basis for interpolation
    step (float): Step size for the executive metric interpolation

    Returns:
    pd.DataFrame: Interpolated DataFrame
    """
    # Get min and max values of the executive metric
    min_val = df[executive_metric].min()
    max_val = df[executive_metric].max()

    # Adjust min and max to be multiples of step
    min_multiple = np.floor(min_val / step) * step
    max_multiple = np.ceil(max_val / step) * step

    # Create new executive metric values with step stride
    new_executive_values = np.arange(min_multiple, max_multiple + step, step)

    # Sort original dataframe by executive metric
    df_sorted = df.sort_values(by=executive_metric)

    # Create result dataframe
    result_df = pd.DataFrame({executive_metric: new_executive_values})

    # Interpolate other numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    other_columns = [col for col in numeric_columns if col != executive_metric]

    for col in other_columns:
        # Use numpy interp for linear interpolation
        interpolated_values = np.interp(
            new_executive_values,
            df_sorted[executive_metric].values,
            df_sorted[col].values,
        )
        result_df[col] = interpolated_values

    return result_df


def expand_dataframes_to_max(dataframes_dict, executive_metric, max_executive_value=None, step=None):
    """
    Expands each DataFrame in the dictionary to the maximum executive metric value using forward-fill.

    Parameters:
    dataframes_dict (dict): Dictionary of DataFrames (e.g., from split_dataframe_by_column + interpolate_dataframe)
    executive_metric (str): Name of the executive metric column
    max_executive_value (float, optional): Maximum executive metric value to expand to.
        If None, uses the maximum value across all dataframes.
    step (float, optional): Step size for extending the executive metric. If None, uses all unique values
        from the input dataframes.

    Returns:
    dict: Dictionary with the same keys, but each DataFrame expanded to max executive metric
        with forward-filled values
    """
    if not dataframes_dict:
        raise ValueError("Input dictionary is empty")

    # Find the maximum executive metric value across all dataframes if not provided
    if max_executive_value is None:
        max_executive_value = max(df[executive_metric].max() for df in dataframes_dict.values())

    # Determine executive metric values based on step parameter
    if step is not None:
        # Find minimum value across all dataframes
        min_executive_value = min(df[executive_metric].min() for df in dataframes_dict.values())

        # Adjust min and max to be multiples of step
        min_multiple = np.floor(min_executive_value / step) * step
        max_multiple = np.ceil(max_executive_value / step) * step

        # Create executive values with step stride using linspace for better floating-point precision
        num_points = int(round((max_multiple - min_multiple) / step)) + 1
        executive_values = np.linspace(min_multiple, max_multiple, num_points).tolist()
    else:
        # Collect all unique executive metric values from all dataframes
        all_executive_values = set()
        for df in dataframes_dict.values():
            all_executive_values.update(df[executive_metric].values)

        # Sort and filter to only include values up to max
        executive_values = sorted([v for v in all_executive_values if v <= max_executive_value])

    # Expand each dataframe
    expanded_dataframes = {}

    for key, df in dataframes_dict.items():
        # Sort dataframe by executive metric for correct forward-fill behavior
        df_sorted = df.sort_values(by=executive_metric).reset_index(drop=True)

        # Create new dataframe with all executive metric values
        expanded_df = pd.DataFrame({executive_metric: executive_values})

        # Get columns from original dataframe (excluding executive_metric)
        other_columns = [col for col in df_sorted.columns if col != executive_metric]

        # For each column, forward fill values
        for col in other_columns:
            col_values = []

            for exec_val in executive_values:
                # Get rows up to and including current executive value
                mask = df_sorted[executive_metric] <= exec_val
                if mask.any():
                    # Forward fill: take the last available value (now correctly sorted)
                    last_value = df_sorted.loc[mask, col].iloc[-1]
                    col_values.append(last_value)
                else:
                    # If no data available yet, use NaN
                    col_values.append(np.nan)

            expanded_df[col] = col_values

        expanded_dataframes[key] = expanded_df

    return expanded_dataframes
