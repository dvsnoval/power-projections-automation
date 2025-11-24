import unittest
import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add the parent directory to the path so we can import from code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.aggregator import interpolate_dataframe, split_dataframe_by_column


class TestInterpolateDataframe:
    def setup_method(self):
        """Set up test data for each test method."""
        self.df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [10.0, 20.0, 30.0, 40.0, 50.0],
                "z": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )

    def test_basic_interpolation(self):
        """Test basic interpolation with step size 0.5."""
        result = interpolate_dataframe(self.df, "x", 0.5)

        # Check that result has correct executive metric values (multiples of step)
        expected_x = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        np.testing.assert_array_equal(result["x"].values, expected_x)

        # Check interpolated values
        assert abs(result[result["x"] == 1.5]["y"].iloc[0] - 15.0) < 1e-7
        assert abs(result[result["x"] == 2.5]["y"].iloc[0] - 25.0) < 1e-7

    def test_step_larger_than_range(self):
        """Test with step size larger than the data range."""
        result = interpolate_dataframe(self.df, "x", 10.0)
        print(result)
        # Should have multiples of step that cover the range
        assert len(result) == 2
        assert result["x"].iloc[0] == 0.0  # floor(1.0/10.0) * 10.0 = 0.0
        assert result["x"].iloc[-1] == 10.0  # ceil(5.0/10.0) * 10.0 = 10.0
        # Check interpolated values at the boundaries
        assert abs(result["y"].iloc[0] - 10.0) < 1e-7  # extrapolated to x=0
        assert abs(result["y"].iloc[-1] - 50.0) < 1e-7  # extrapolated to x=10

    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        single_df = pd.DataFrame({"x": [5.0], "y": [25.0]})
        result = interpolate_dataframe(single_df, "x", 1.0)

        # With single value 5.0 and step 1.0, we get one point at x=5.0
        assert len(result) == 1
        assert result["x"].iloc[0] == 5.0
        assert result["y"].iloc[0] == 25.0

    def test_unsorted_data(self):
        """Test with unsorted executive metric data."""
        unsorted_df = pd.DataFrame({"x": [3.0, 1.0, 5.0, 2.0, 4.0], "y": [30.0, 10.0, 50.0, 20.0, 40.0]})
        result = interpolate_dataframe(unsorted_df, "x", 1.0)

        expected_x = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected_y = [10.0, 20.0, 30.0, 40.0, 50.0]

        np.testing.assert_array_equal(result["x"].values, expected_x)
        np.testing.assert_array_equal(result["y"].values, expected_y)

    def test_non_numeric_columns_ignored(self):
        """Test that non-numeric columns are ignored."""
        mixed_df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0], "label": ["A", "B", "C"]})
        result = interpolate_dataframe(mixed_df, "x", 0.5)

        # Should only have numeric columns
        assert "x" in result.columns
        assert "y" in result.columns
        assert "label" not in result.columns

    def test_exact_step_boundaries(self):
        """Test interpolation with exact step boundaries."""
        result = interpolate_dataframe(self.df, "x", 1.0)

        # Should match original data exactly
        expected_x = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected_y = [10.0, 20.0, 30.0, 40.0, 50.0]

        np.testing.assert_array_equal(result["x"].values, expected_x)
        np.testing.assert_array_equal(result["y"].values, expected_y)

    def test_small_step_size(self):
        """Test with very small step size."""
        result = interpolate_dataframe(self.df, "x", 0.1)

        # Should have many interpolated points
        assert len(result) > len(self.df)

        # Check that interpolation maintains linearity
        assert abs(result[result["x"] == 1.1]["y"].iloc[0] - 11.0) < 1e-5

    def test_fractional_boundaries(self):
        """Test with fractional min/max values to ensure proper multiples."""
        frac_df = pd.DataFrame({"x": [1.3, 2.7, 3.9], "y": [13.0, 27.0, 39.0]})
        result = interpolate_dataframe(frac_df, "x", 0.5)

        # Should extend to proper multiples: floor(1.3/0.5)*0.5=1.0, ceil(3.9/0.5)*0.5=4.0
        expected_x = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        np.testing.assert_array_equal(result["x"].values, expected_x)

        # Check that all x values are multiples of step
        for x_val in result["x"].values:
            assert abs(x_val % 0.5) < 1e-10, f"Value {x_val} is not a multiple of 0.5"


class TestSplitDataframeByColumn:
    def setup_method(self):
        """Set up test data for each test method."""
        self.df = pd.DataFrame(
            {
                "category": ["A", "B", "A", "C", "B", "A", "C"],
                "value": [1, 2, 3, 4, 5, 6, 7],
                "score": [10.5, 20.1, 30.2, 40.3, 50.4, 60.5, 70.6],
            }
        )

    def test_basic_split_by_category(self):
        """Test basic splitting by categorical column."""
        result = split_dataframe_by_column(self.df, "category")

        # Should have 3 dataframes (A, B, C)
        assert len(result) == 3

        # Check each dataframe has constant category value
        categories_found = set()
        for df_subset in result:
            unique_categories = df_subset["category"].unique()
            assert len(unique_categories) == 1
            categories_found.add(unique_categories[0])

        # Should have found all categories
        assert categories_found == {"A", "B", "C"}

    def test_split_preserves_data(self):
        """Test that splitting preserves all original data."""
        result = split_dataframe_by_column(self.df, "category")

        # Concatenate all split dataframes
        combined = pd.concat(result, ignore_index=True)

        # Should have same number of rows
        assert len(combined) == len(self.df)

        # Should have same columns
        assert set(combined.columns) == set(self.df.columns)

        # Check specific subsets
        df_a = next(df for df in result if df["category"].iloc[0] == "A")
        df_b = next(df for df in result if df["category"].iloc[0] == "B")
        df_c = next(df for df in result if df["category"].iloc[0] == "C")

        assert len(df_a) == 3  # A appears 3 times
        assert len(df_b) == 2  # B appears 2 times
        assert len(df_c) == 2  # C appears 2 times

        # Check values are preserved
        np.testing.assert_array_equal(sorted(df_a["value"].values), [1, 3, 6])
        np.testing.assert_array_equal(sorted(df_b["value"].values), [2, 5])
        np.testing.assert_array_equal(sorted(df_c["value"].values), [4, 7])

    def test_split_by_numeric_column(self):
        """Test splitting by numeric column."""
        numeric_df = pd.DataFrame({"group": [1, 2, 1, 3, 2, 1], "data": ["a", "b", "c", "d", "e", "f"]})

        result = split_dataframe_by_column(numeric_df, "group")

        assert len(result) == 3  # Groups 1, 2, 3

        # Verify each group
        for df_subset in result:
            group_val = df_subset["group"].iloc[0]
            assert all(df_subset["group"] == group_val)

    def test_single_unique_value(self):
        """Test with column having only one unique value."""
        single_df = pd.DataFrame({"constant": ["same", "same", "same"], "varying": [1, 2, 3]})

        result = split_dataframe_by_column(single_df, "constant")

        assert len(result) == 1
        assert len(result[0]) == 3
        assert all(result[0]["constant"] == "same")

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame({"col": []})
        result = split_dataframe_by_column(empty_df, "col")

        assert len(result) == 0

    def test_invalid_column_name(self):
        """Test with non-existent column name."""
        with pytest.raises(ValueError, match="Column 'nonexistent' not found in DataFrame"):
            split_dataframe_by_column(self.df, "nonexistent")

    def test_dataframe_independence(self):
        """Test that returned dataframes are independent copies."""
        result = split_dataframe_by_column(self.df, "category")

        # Modify one of the returned dataframes
        df_subset = result[0]
        original_value = df_subset["value"].iloc[0]
        df_subset.loc[df_subset.index[0], "value"] = 999

        # Original dataframe should be unchanged
        assert self.df["value"].iloc[0] != 999
        assert self.df["value"].iloc[0] == original_value

    def test_with_nan_values(self):
        """Test splitting when column contains NaN values."""
        nan_df = pd.DataFrame({"category": ["A", np.nan, "A", "B", np.nan], "value": [1, 2, 3, 4, 5]})

        result = split_dataframe_by_column(nan_df, "category")

        # Should have 3 groups: 'A', 'B', and NaN
        assert len(result) == 3

        # Find the NaN group and other groups
        nan_group = None
        a_group = None
        b_group = None

        for df_subset in result:
            if len(df_subset) > 0:  # Ensure dataframe is not empty
                first_value = df_subset["category"].iloc[0]
                if pd.isna(first_value):
                    nan_group = df_subset
                elif first_value == "A":
                    a_group = df_subset
                elif first_value == "B":
                    b_group = df_subset

        # Verify all groups were found
        assert nan_group is not None
        assert a_group is not None
        assert b_group is not None

        # Check group sizes and content
        assert len(nan_group) == 2  # Two NaN values
        assert len(a_group) == 2  # Two 'A' values
        assert len(b_group) == 1  # One 'B' value

        # Verify all NaN values are in nan_group
        assert all(pd.isna(nan_group["category"]))

        # Verify other groups have correct values
        assert all(a_group["category"] == "A")
        assert all(b_group["category"] == "B")


if __name__ == "__main__":
    unittest.main()
