import pandas as pd
import numpy as np

def split_dataframe_by_column(df, column_name):
    """
    Splits a DataFrame into a list of DataFrames, each with constant values in the specified column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to split
    column_name (str): Name of the column to use for splitting
    
    Returns:
    list[pd.DataFrame]: List of DataFrames, each containing rows with the same value in column_name
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    # Get unique values in the specified column
    unique_values = df[column_name].unique()
    
    # Create list of dataframes, one for each unique value
    dataframes = []
    for value in unique_values:
        if pd.isna(value):
            # Handle NaN values separately since NaN != NaN
            subset_df = df[df[column_name].isna()].copy()
        else:
            subset_df = df[df[column_name] == value].copy()
        
        if len(subset_df) > 0:  # Only add non-empty dataframes
            dataframes.append(subset_df)
    
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
            df_sorted[col].values
        )
        result_df[col] = interpolated_values
    
    return result_df