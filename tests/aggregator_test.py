import unittest
import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add the parent directory to the path so we can import from utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.aggregator import interpolate_dataframe, split_dataframe_by_column, expand_dataframes_to_max


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
        for key, df_subset in result.items():
            unique_categories = df_subset["category"].unique()
            assert len(unique_categories) == 1
            categories_found.add(unique_categories[0])

        # Should have found all categories
        assert categories_found == {"A", "B", "C"}

    def test_split_preserves_data(self):
        """Test that splitting preserves all original data."""
        result = split_dataframe_by_column(self.df, "category")

        # Concatenate all split dataframes
        combined = pd.concat(result.values(), ignore_index=True)

        # Should have same number of rows
        assert len(combined) == len(self.df)

        # Should have same columns
        assert set(combined.columns) == set(self.df.columns)

        # Check specific subsets
        df_a = result["A"]
        df_b = result["B"]
        df_c = result["C"]

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
        assert 1 in result
        assert 2 in result
        assert 3 in result

        # Verify each group
        for group_val, df_subset in result.items():
            assert all(df_subset["group"] == group_val)

    def test_single_unique_value(self):
        """Test with column having only one unique value."""
        single_df = pd.DataFrame({"constant": ["same", "same", "same"], "varying": [1, 2, 3]})

        result = split_dataframe_by_column(single_df, "constant")

        assert len(result) == 1
        assert "same" in result
        assert len(result["same"]) == 3
        assert all(result["same"]["constant"] == "same")

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
        df_subset = result["A"]
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
        assert "A" in result
        assert "B" in result
        assert "NaN" in result

        nan_group = result["NaN"]
        a_group = result["A"]
        b_group = result["B"]

        # Check group sizes and content
        assert len(nan_group) == 2  # Two NaN values
        assert len(a_group) == 2  # Two 'A' values
        assert len(b_group) == 1  # One 'B' value

        # Verify all NaN values are in nan_group
        assert all(pd.isna(nan_group["category"]))

        # Verify other groups have correct values
        assert all(a_group["category"] == "A")
        assert all(b_group["category"] == "B")


class TestExpandDataframesToMax:
    def setup_method(self):
        """Set up test data for each test method."""
        # Create sample dataframes with different ranges
        self.df1 = pd.DataFrame(
            {"time": [0.0, 1.0, 2.0], "value_a": [10.0, 20.0, 30.0], "value_b": [100.0, 200.0, 300.0]}
        )

        self.df2 = pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "value_a": [5.0, 15.0, 25.0, 35.0, 45.0],
                "value_b": [50.0, 150.0, 250.0, 350.0, 450.0],
            }
        )

        self.df3 = pd.DataFrame({"time": [1.0, 2.0, 3.0], "value_a": [12.0, 22.0, 32.0]})

        self.dataframes_dict = {"group1": self.df1, "group2": self.df2, "group3": self.df3}

    def test_basic_expansion(self):
        """Test basic expansion to maximum executive metric value."""
        result = expand_dataframes_to_max(self.dataframes_dict, "time")

        # All dataframes should have same time values (all unique time values up to max)
        expected_times = [0.0, 1.0, 2.0, 3.0, 4.0]
        for key, df in result.items():
            np.testing.assert_array_equal(df["time"].values, expected_times)

        # Check that original data is preserved
        # group1 original data at time 0, 1, 2
        assert result["group1"]["value_a"].iloc[0] == 10.0  # time=0
        assert result["group1"]["value_a"].iloc[1] == 20.0  # time=1
        assert result["group1"]["value_a"].iloc[2] == 30.0  # time=2

        # group2 should have all its original values
        assert result["group2"]["value_a"].iloc[0] == 5.0  # time=0
        assert result["group2"]["value_a"].iloc[4] == 45.0  # time=4

    def test_forward_fill_behavior(self):
        """Test that values are forward-filled correctly."""
        result = expand_dataframes_to_max(self.dataframes_dict, "time")

        # group1 only has data up to time=2, so time=3 and time=4 should be forward-filled
        group1_result = result["group1"]
        assert group1_result["value_a"].iloc[2] == 30.0  # time=2 (last original)
        assert group1_result["value_a"].iloc[3] == 30.0  # time=3 (forward-filled)
        assert group1_result["value_a"].iloc[4] == 30.0  # time=4 (forward-filled)

        assert group1_result["value_b"].iloc[2] == 300.0  # time=2 (last original)
        assert group1_result["value_b"].iloc[3] == 300.0  # time=3 (forward-filled)
        assert group1_result["value_b"].iloc[4] == 300.0  # time=4 (forward-filled)

    def test_dataframes_with_gaps(self):
        """Test with dataframe that starts after time=0."""
        result = expand_dataframes_to_max(self.dataframes_dict, "time")

        # group3 starts at time=1, so time=0 should be NaN
        group3_result = result["group3"]
        assert pd.isna(group3_result["value_a"].iloc[0])  # time=0 (NaN)
        assert group3_result["value_a"].iloc[1] == 12.0  # time=1 (original)
        assert group3_result["value_a"].iloc[2] == 22.0  # time=2 (original)
        assert group3_result["value_a"].iloc[3] == 32.0  # time=3 (original)
        assert group3_result["value_a"].iloc[4] == 32.0  # time=4 (forward-filled)

    def test_custom_max_value(self):
        """Test with custom maximum executive metric value."""
        result = expand_dataframes_to_max(self.dataframes_dict, "time", max_executive_value=2.0)

        # All dataframes should only have times up to 2.0
        expected_times = [0.0, 1.0, 2.0]
        for key, df in result.items():
            np.testing.assert_array_equal(df["time"].values, expected_times)
            assert len(df) == 3

    def test_single_dataframe(self):
        """Test with dictionary containing single dataframe."""
        single_dict = {"only": self.df1}
        result = expand_dataframes_to_max(single_dict, "time")

        assert len(result) == 1
        assert "only" in result
        # Should have same times as original since it's the only one
        np.testing.assert_array_equal(result["only"]["time"].values, self.df1["time"].values)

    def test_empty_dictionary(self):
        """Test with empty dictionary."""
        with pytest.raises(ValueError, match="Input dictionary is empty"):
            expand_dataframes_to_max({}, "time")

    def test_all_dataframes_same_range(self):
        """Test with all dataframes having the same range."""
        same_range_dict = {
            "a": pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]}),
            "b": pd.DataFrame({"x": [1.0, 2.0, 3.0], "z": [15.0, 25.0, 35.0]}),
        }

        result = expand_dataframes_to_max(same_range_dict, "x")

        # Both should have same x values
        for key, df in result.items():
            np.testing.assert_array_equal(df["x"].values, [1.0, 2.0, 3.0])

    def test_different_columns_per_dataframe(self):
        """Test that dataframes with different columns are handled correctly."""
        mixed_cols_dict = {
            "df1": pd.DataFrame({"time": [1.0, 2.0], "col_a": [10.0, 20.0], "col_b": [100.0, 200.0]}),
            "df2": pd.DataFrame({"time": [1.0, 2.0, 3.0], "col_c": [5.0, 15.0, 25.0]}),
            "df3": pd.DataFrame({"time": [1.0, 2.0, 3.0], "col_a": [8.0, 18.0, 28.0], "col_d": [80.0, 180.0, 280.0]}),
        }

        result = expand_dataframes_to_max(mixed_cols_dict, "time")

        # Check that each dataframe retains its own columns
        assert set(result["df1"].columns) == {"time", "col_a", "col_b"}
        assert set(result["df2"].columns) == {"time", "col_c"}
        assert set(result["df3"].columns) == {"time", "col_a", "col_d"}

        # All should have time up to 3.0
        for df in result.values():
            assert df["time"].max() == 3.0

    def test_non_overlapping_ranges(self):
        """Test with dataframes that have non-overlapping time ranges."""
        non_overlap_dict = {
            "early": pd.DataFrame({"time": [1.0, 2.0], "value": [10.0, 20.0]}),
            "late": pd.DataFrame({"time": [5.0, 6.0], "value": [50.0, 60.0]}),
        }

        result = expand_dataframes_to_max(non_overlap_dict, "time")

        # Should have times 1, 2, 5, 6
        expected_times = [1.0, 2.0, 5.0, 6.0]
        for df in result.values():
            np.testing.assert_array_equal(df["time"].values, expected_times)

        # Early dataframe should have NaN for times 5 and 6 initially, then forward fill from time 2
        early_result = result["early"]
        assert early_result["value"].iloc[0] == 10.0  # time=1
        assert early_result["value"].iloc[1] == 20.0  # time=2
        assert early_result["value"].iloc[2] == 20.0  # time=5 (forward-filled)
        assert early_result["value"].iloc[3] == 20.0  # time=6 (forward-filled)

        # Late dataframe should have NaN for times 1 and 2
        late_result = result["late"]
        assert pd.isna(late_result["value"].iloc[0])  # time=1 (NaN)
        assert pd.isna(late_result["value"].iloc[1])  # time=2 (NaN)
        assert late_result["value"].iloc[2] == 50.0  # time=5
        assert late_result["value"].iloc[3] == 60.0  # time=6

    def test_preserve_dictionary_keys(self):
        """Test that dictionary keys are preserved in output."""
        result = expand_dataframes_to_max(self.dataframes_dict, "time")

        assert set(result.keys()) == set(self.dataframes_dict.keys())
        assert "group1" in result
        assert "group2" in result
        assert "group3" in result

    def test_with_unsorted_times(self):
        """Test with unsorted time values in individual dataframes."""
        unsorted_dict = {
            "unsorted": pd.DataFrame({"time": [3.0, 1.0, 2.0], "value": [30.0, 10.0, 20.0]}),
            "sorted": pd.DataFrame({"time": [1.0, 2.0, 3.0, 4.0], "value": [15.0, 25.0, 35.0, 45.0]}),
        }

        result = expand_dataframes_to_max(unsorted_dict, "time")

        # Should still work correctly with forward fill
        unsorted_result = result["unsorted"]
        expected_times = [1.0, 2.0, 3.0, 4.0]
        np.testing.assert_array_equal(unsorted_result["time"].values, expected_times)

        # Check forward fill works (value at time=1 is 10, forward filled to time=4 becomes 30)
        assert unsorted_result["value"].iloc[0] == 10.0  # time=1
        assert unsorted_result["value"].iloc[1] == 20.0  # time=2
        assert unsorted_result["value"].iloc[2] == 30.0  # time=3
        assert unsorted_result["value"].iloc[3] == 30.0  # time=4 (forward-filled)

    def test_multiple_columns_forward_fill_independently(self):
        """Test that multiple columns are forward-filled independently."""
        test_dict = {
            "test": pd.DataFrame(
                {"time": [1.0, 2.0], "col1": [10.0, 20.0], "col2": [100.0, 200.0], "col3": [1000.0, 2000.0]}
            ),
            "reference": pd.DataFrame({"time": [1.0, 2.0, 3.0, 4.0], "dummy": [0.0, 0.0, 0.0, 0.0]}),
        }

        result = expand_dataframes_to_max(test_dict, "time")

        test_result = result["test"]
        # All columns should be forward-filled to the last value
        assert test_result["col1"].iloc[-1] == 20.0
        assert test_result["col2"].iloc[-1] == 200.0
        assert test_result["col3"].iloc[-1] == 2000.0

    def test_with_step_parameter(self):
        """Test expansion with step parameter to create evenly-spaced values."""
        test_dict = {
            "df1": pd.DataFrame({"time": [1.0, 3.5], "value": [10.0, 35.0]}),
            "df2": pd.DataFrame({"time": [0.5, 2.5, 4.5], "value": [5.0, 25.0, 45.0]}),
        }

        result = expand_dataframes_to_max(test_dict, "time", step=1.0)

        # With step=1.0, should have times: 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
        # (floor(0.5/1.0)*1.0 = 0.0 to ceil(4.5/1.0)*1.0 = 5.0)
        expected_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        for df in result.values():
            np.testing.assert_array_equal(df["time"].values, expected_times)

        # Check forward fill for df1
        df1_result = result["df1"]
        assert pd.isna(df1_result["value"].iloc[0])  # time=0.0 (no data yet)
        assert df1_result["value"].iloc[1] == 10.0  # time=1.0
        assert df1_result["value"].iloc[2] == 10.0  # time=2.0 (forward-filled from 1.0)
        assert df1_result["value"].iloc[3] == 10.0  # time=3.0 (forward-filled from 1.0, 3.5 is in future)
        assert df1_result["value"].iloc[4] == 35.0  # time=4.0 (forward-filled from 3.5)
        assert df1_result["value"].iloc[5] == 35.0  # time=5.0 (forward-filled from 3.5)

    def test_with_step_parameter_smaller_than_gaps(self):
        """Test step parameter with smaller step size than data gaps."""
        test_dict = {
            "test": pd.DataFrame({"x": [1.0, 5.0], "y": [100.0, 500.0]}),
        }

        result = expand_dataframes_to_max(test_dict, "x", step=0.5)

        # Should have x values from 1.0 to 5.0 with 0.5 step
        expected_x = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        np.testing.assert_array_equal(result["test"]["x"].values, expected_x)

        # All values between 1.0 and 5.0 should be forward-filled
        test_result = result["test"]
        assert test_result["y"].iloc[0] == 100.0  # x=1.0
        assert test_result["y"].iloc[1] == 100.0  # x=1.5 (forward-filled)
        assert test_result["y"].iloc[4] == 100.0  # x=3.0 (forward-filled)
        assert test_result["y"].iloc[-1] == 500.0  # x=5.0

    def test_with_step_and_custom_max(self):
        """Test step parameter combined with custom max_executive_value."""
        test_dict = {
            "df1": pd.DataFrame({"time": [1.0, 2.0, 3.0, 4.0], "value": [10.0, 20.0, 30.0, 40.0]}),
        }

        result = expand_dataframes_to_max(test_dict, "time", max_executive_value=2.5, step=0.5)

        # With max=2.5 and step=0.5, should go from floor(1.0/0.5)*0.5=1.0 to ceil(2.5/0.5)*0.5=2.5
        expected_times = [1.0, 1.5, 2.0, 2.5]
        np.testing.assert_array_equal(result["df1"]["time"].values, expected_times)

        df1_result = result["df1"]
        assert df1_result["value"].iloc[0] == 10.0  # time=1.0
        assert df1_result["value"].iloc[1] == 10.0  # time=1.5 (forward-filled)
        assert df1_result["value"].iloc[2] == 20.0  # time=2.0
        assert df1_result["value"].iloc[3] == 20.0  # time=2.5 (forward-filled)

    def test_step_with_multiple_dataframes(self):
        """Test that step parameter works correctly with multiple dataframes."""
        test_dict = {
            "early": pd.DataFrame({"time": [0.0, 1.0], "value": [0.0, 10.0]}),
            "late": pd.DataFrame({"time": [2.0, 3.0], "value": [20.0, 30.0]}),
        }

        result = expand_dataframes_to_max(test_dict, "time", step=1.0)

        # Should have times from 0 to 3 with step 1
        expected_times = [0.0, 1.0, 2.0, 3.0]
        for df in result.values():
            np.testing.assert_array_equal(df["time"].values, expected_times)

        # Early dataframe should forward-fill after time=1
        early_result = result["early"]
        assert early_result["value"].iloc[0] == 0.0  # time=0.0
        assert early_result["value"].iloc[1] == 10.0  # time=1.0
        assert early_result["value"].iloc[2] == 10.0  # time=2.0 (forward-filled)
        assert early_result["value"].iloc[3] == 10.0  # time=3.0 (forward-filled)

        # Late dataframe should have NaN before time=2
        late_result = result["late"]
        assert pd.isna(late_result["value"].iloc[0])  # time=0.0 (no data)
        assert pd.isna(late_result["value"].iloc[1])  # time=1.0 (no data)
        assert late_result["value"].iloc[2] == 20.0  # time=2.0
        assert late_result["value"].iloc[3] == 30.0  # time=3.0

    def test_step_none_uses_original_behavior(self):
        """Test that step=None uses the original behavior of collecting unique values."""
        test_dict = {
            "df1": pd.DataFrame({"time": [1.0, 3.7], "value": [10.0, 37.0]}),
            "df2": pd.DataFrame({"time": [2.1, 4.9], "value": [21.0, 49.0]}),
        }

        result = expand_dataframes_to_max(test_dict, "time", step=None)

        # Should have all unique time values: 1.0, 2.1, 3.7, 4.9
        expected_times = [1.0, 2.1, 3.7, 4.9]
        for df in result.values():
            np.testing.assert_array_equal(df["time"].values, expected_times)


if __name__ == "__main__":
    unittest.main()
