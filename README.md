# Aggreg

A Python library for data aggregation and interpolation utilities.

## Features

- **DataFrame Interpolation**: Interpolate DataFrames with guaranteed step multiples
- **DataFrame Splitting**: Split DataFrames by column values with proper NaN handling
- **Robust Testing**: Comprehensive test suite with edge case coverage

## Installation

### Using uv (recommended)

```bash
# Install uv if you haven't already
pip install uv

# Install the project
uv pip install -e .
```

### Using pip

```bash
pip install -e .
```

## Development Setup

### Using uv

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

## Usage

### DataFrame Interpolation

```python
import pandas as pd
from code.aggregator import interpolate_dataframe

# Create sample data
df = pd.DataFrame({
    'x': [1.0, 2.0, 3.0, 4.0, 5.0],
    'y': [10.0, 20.0, 30.0, 40.0, 50.0]
})

# Interpolate with step size 0.5
result = interpolate_dataframe(df, 'x', 0.5)
print(result)
```

### DataFrame Splitting

```python
import pandas as pd
from code.aggregator import split_dataframe_by_column

# Create sample data
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B'],
    'value': [1, 2, 3, 4, 5]
})

# Split by category
dataframes = split_dataframe_by_column(df, 'category')
print(f"Split into {len(dataframes)} DataFrames")
```

## Running Tests

### Using pytest

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=code --cov-report=html

# Run specific test file
pytest tests/aggregator_test.py

# Run specific test
pytest tests/aggregator_test.py::TestInterpolateDataframe::test_basic_interpolation
```

### Using uv

```bash
# Run tests with uv
uv run pytest

# Run with coverage
uv run pytest --cov=code --cov-report=html
```

## Code Quality Tools

### Formatting

```bash
# Format code with black
black code tests

# Sort imports with isort
isort code tests
```

### Linting

```bash
# Check with flake8
flake8 code tests

# Type checking with mypy
mypy code
```

### All quality checks

```bash
# Run all quality checks
black --check code tests
isort --check-only code tests
flake8 code tests
mypy code
pytest
```

## Project Structure

```
aggreg/
├── code/
│   ├── __init__.py
│   └── aggregator.py       # Main functions
├── tests/
│   ├── __init__.py
│   ├── aggregator_test.py  # Test suite
│   └── conftest.py
├── pyproject.toml          # Project configuration
├── README.md
└── pytest.ini             # Legacy pytest config
```

## API Reference

### `interpolate_dataframe(df, executive_metric, step)`

Interpolates a DataFrame based on an executive metric column with guaranteed step multiples.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame with numeric columns
- `executive_metric` (str): Name of the column to use as the basis for interpolation
- `step` (float): Step size for the executive metric interpolation

**Returns:**
- `pd.DataFrame`: Interpolated DataFrame with executive metric values as multiples of step

### `split_dataframe_by_column(df, column_name)`

Splits a DataFrame into a list of DataFrames, each with constant values in the specified column.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame to split
- `column_name` (str): Name of the column to use for splitting

**Returns:**
- `list[pd.DataFrame]`: List of DataFrames, each containing rows with the same value in column_name

## License

This project is licensed under the MIT License.