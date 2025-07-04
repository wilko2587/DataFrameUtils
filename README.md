# DataFrameUtils - Weighted Bin Calculator

A Python utility for calculating weighted averages of quantity2 values in bins of quantity1 increments, with support for temporal data and grouped analysis.

## Overview

This tool is designed for financial and time-series data analysis where you need to:
- Group data by multiple identifiers (e.g., asset and type)
- Calculate weighted averages of one quantity based on another quantity
- Handle events that span multiple bins
- Process data chronologically (only future events are considered)

## Use Cases

- **Trading Data**: Calculate weighted average prices in volume bins
- **Inventory Analysis**: Analyze costs across order size ranges
- **Time Series**: Study value distributions across duration bins
- **Financial Modeling**: Risk analysis across position size categories

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
    'Asset': ['A', 'A', 'A', 'A'],
    'Type': ['X', 'X', 'X', 'X'],
    'Time': pd.date_range('2024-01-01', periods=4, freq='h'),
    'Volume': [50, 75, 100, 25],
    'Price': [10, 20, 30, 40]
})

# Calculate weighted bins
result = calculate_weighted_bins(
    df=df,
    id1_col='Asset',
    id2_col='Type',
    timestamp_col='Time',
    q1_col='Volume',
    q2_col='Price',
    bin_size=100,
    max_bins=3
)

print(result)
```

## How It Works

### Algorithm Overview

1. **Grouping**: Data is grouped by ID1 and ID2 (e.g., Asset and Type)
2. **Sorting**: Within each group, data is sorted chronologically by timestamp
3. **Processing**: For each row, the algorithm:
   - Looks at future events (after the current timestamp)
   - Creates bins based on quantity1 values
   - Calculates weighted averages of quantity2 within each bin
   - Handles events that span multiple bins by splitting them proportionally

### Example Calculation

For a row with future events:
- Event 1: Volume=199, Price=39
- Event 2: Volume=71, Price=28
- Event 3: Volume=61, Price=50

With bin_size=50, the results would be:
- **Bin 1 (0-50)**: 50×39 = 39.00 average
- **Bin 2 (50-100)**: 50×39 = 39.00 average  
- **Bin 3 (100-150)**: 50×39 = 39.00 average
- **Bin 4 (150-200)**: 49×39 + 1×28 = 38.78 average
- **Bin 5 (200-250)**: 50×28 = 28.00 average

## API Reference

### `calculate_weighted_bins()`

**Parameters:**
- `df` (DataFrame): Input data
- `id1_col` (str): First identifier column (default: 'ID1')
- `id2_col` (str): Second identifier column (default: 'ID2')
- `timestamp_col` (str): Timestamp column (default: 'timestamp')
- `q1_col` (str): Quantity1 column for binning (default: 'quantity1')
- `q2_col` (str): Quantity2 column for averaging (default: 'quantity2')
- `bin_size` (float): Size of each bin (default: 100)
- `max_bins` (int): Maximum number of bins to calculate (default: 10)

**Returns:**
- DataFrame with original columns plus `bin_1_avg`, `bin_2_avg`, etc.

## Data Requirements

Your DataFrame must contain:
- **ID columns**: Two identifier columns for grouping
- **Timestamp column**: For chronological ordering
- **Quantity columns**: Two numeric columns (one for binning, one for averaging)

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
    id1_col='Asset',
    id2_col='Type',
    timestamp_col='Time',
    q1_col='Volume',
    q2_col='Price',
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