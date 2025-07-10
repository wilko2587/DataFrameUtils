# DataFrameUtils - Weighted Bin Calculator

A Python utility for calculating weighted averages of quantity2 values in bins of quantity1 increments, with support for temporal data and grouped analysis.

## Overview

This tool is designed for time-series data analysis where you need to:
- Group data by identifiers
- Calculate weighted averages of one quantity based on another quantity
- Handle events that span multiple bins
- Process data chronologically (only future events are considered)

## Use Cases

- **Data Analysis**: Calculate weighted averages across different size ranges
- **Inventory Analysis**: Analyze costs across order size ranges
- **Time Series**: Study value distributions across duration bins
- **Statistical Modeling**: Analysis across different quantity categories

## Installation

```bash
pip install pandas numpy tqdm
```

## Quick Start

```python
from weighted_bin_calculator import calculate_weighted_bins
import pandas as pd

# Create sample data
df = pd.DataFrame({
    'ID1': ['A', 'A', 'A', 'A'],
    'timestamp': pd.date_range('2024-01-01', periods=4, freq='h'),
    'quantity1': [50, 75, 100, 25],
    'quantity2': [10, 20, 30, 40]
})

# Calculate weighted bins
result = calculate_weighted_bins(
    df=df,
    id1_col='ID1',
    timestamp_col='timestamp',
    q1_col='quantity1',
    q2_col='quantity2',
    bin_size=100,
    max_bins=3
)

print(result)
```

## How It Works

### Algorithm Overview

1. **Grouping**: Data is grouped by ID1 (e.g., entity identifier)
2. **Sorting**: Within each group, data is sorted chronologically by timestamp
3. **Processing**: For each row, the algorithm:
   - Looks at future events (after the current timestamp)
   - Creates bins based on quantity1 values
   - Calculates weighted averages of quantity2 within each bin
   - Handles events that span multiple bins by splitting them proportionally

### Example Calculation

For a row with future events:
- Event 1: quantity1=199, quantity2=39
- Event 2: quantity1=71, quantity2=28
- Event 3: quantity1=61, quantity2=50

With bin_size=50, the results would be:
- **Bin 1 (0-50)**: 36×45 + 14×39 = 43.32 average
- **Bin 2 (50-100)**: 50×39 = 39.00 average  
- **Bin 3 (100-150)**: 50×39 = 39.00 average
- **Bin 4 (150-200)**: 50×39 = 39.00 average
- **Bin 5 (200-250)**: 35×39 + 15×28 = 35.70 average

## API Reference

### `calculate_weighted_bins()`

**Parameters:**
- `df` (DataFrame): Input data
- `id1_col` (str): Identifier column for grouping (default: 'ID1')
- `timestamp_col` (str): Timestamp column (default: 'timestamp')
- `q1_col` (str): Quantity1 column for binning (default: 'quantity1')
- `q2_col` (str): Quantity2 column for averaging (default: 'quantity2')
- `bin_size` (float): Size of each bin (default: 100)
- `max_bins` (int): Maximum number of bins to calculate (default: 10)

**Returns:**
- DataFrame with original columns plus `bin_1_avg`, `bin_2_avg`, etc.

## Data Requirements

Your DataFrame must contain:
- **ID1 column**: Identifier used for grouping
- **Timestamp column**: For chronological ordering
- **Quantity1 column**: Numeric column used for binning
- **Quantity2 column**: Numeric column used for weighted averaging

## Examples

### Basic Usage
```python
# Use default column names
result = calculate_weighted_bins(df, bin_size=50, max_bins=5)
```

### Custom Column Names
```python
# Use your own column names
result = calculate_weighted_bins(
    df=df,
    id1_col='Entity',
    timestamp_col='Time',
    q1_col='Size',
    q2_col='Value',
    bin_size=100,
    max_bins=3
)
```

### Generate Sample Data
```python
from generate_data import generate_sample_data

# Generate 1000 rows of sample data
df = generate_sample_data(n_rows=1000, n_id1s=5, n_id2s=3)
```

## Performance

The algorithm is optimized for:
- **Large datasets**: Uses numpy arrays for fast computation
- **Memory efficiency**: Processes data in chunks
- **Progress tracking**: Shows progress bars for long operations

Typical performance:
- 100,000 rows: ~2-5 seconds
- 1,000,000 rows: ~20-50 seconds

## Testing

Run the comprehensive test:
```bash
python weighted_bin_calculator.py
```

This will:
1. Run a demonstration with custom column names
2. Execute a comprehensive test with detailed bin transitions
3. Save results to `weighted_bin_results.csv`

## File Structure

```
DataFrameUtils/
├── weighted_bin_calculator.py  # Main function and tests
├── generate_data.py            # Data generation utilities
├── README.md                   # This file
├── sample_data.csv            # Generated sample data (optional)
└── weighted_bin_results.csv   # Test results (generated)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is open source and available under the MIT License. 