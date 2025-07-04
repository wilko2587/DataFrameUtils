import pandas as pd
import numpy as np
import time
from tqdm import tqdm

def calculate_weighted_bins(df, id1_col='ID1', id2_col='ID2', timestamp_col='timestamp', 
                           q1_col='quantity1', q2_col='quantity2', bin_size=100, max_bins=10):
    """
    Calculate weighted averages of quantity2 in bins of quantity1 increments.
    
    This function groups data by ID1 and ID2, then for each row calculates weighted averages
    of quantity2 in bins of quantity1 increments. Events that span bin boundaries are handled
    by splitting them proportionally.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the data
    id1_col : str
        Column name for the first identifier (default: 'ID1')
    id2_col : str
        Column name for the second identifier (default: 'ID2')
    timestamp_col : str
        Column name for timestamps (default: 'timestamp')
    q1_col : str
        Column name for quantity1 values (default: 'quantity1')
    q2_col : str
        Column name for quantity2 values (default: 'quantity2')
    bin_size : float
        Size of each quantity1 bin (default: 100)
    max_bins : int
        Maximum number of bins to calculate (default: 10)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with original columns plus bin average columns (bin_1_avg, bin_2_avg, etc.)
    
    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'ID1': ['A', 'A', 'A', 'A'],
    ...     'ID2': ['x', 'x', 'x', 'x'],
    ...     'timestamp': pd.date_range('2024-01-01', periods=4, freq='H'),
    ...     'quantity1': [50, 75, 100, 25],
    ...     'quantity2': [10, 20, 30, 40]
    ... })
    >>> result = calculate_weighted_bins(df)
    >>> print(result)
    """
    
    # Start timing
    start_time = time.time()
    
    # Validate input columns exist
    required_cols = [id1_col, id2_col, timestamp_col, q1_col, q2_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Sort by ID1, ID2, and timestamp
    df_sorted = df.sort_values([id1_col, id2_col, timestamp_col]).reset_index(drop=True)
    
    # Pre-compute group indices for O(1) lookup
    print("Pre-computing group indices...")
    group_indices = {}
    for idx, row in df_sorted.iterrows():
        key = (row[id1_col], row[id2_col])
        if key not in group_indices:
            group_indices[key] = []
        group_indices[key].append(idx)
    
    # Convert to numpy arrays for faster access
    id1_arr = df_sorted[id1_col].values
    id2_arr = df_sorted[id2_col].values
    timestamp_arr = df_sorted[timestamp_col].values
    q1_arr = df_sorted[q1_col].values
    q2_arr = df_sorted[q2_col].values
    
    # Initialize result list
    results = []
    n = len(df_sorted)
    
    print("Processing rows with optimized algorithm...")
    # Process each row with progress bar
    for idx in tqdm(range(n), desc="Processing rows", unit="rows"):
        current_row = df_sorted.iloc[idx]
        key = (current_row[id1_col], current_row[id2_col])
        
        # Get indices for this group
        group_idx_list = group_indices[key]
        
        # Find position of current row in group
        current_pos = group_idx_list.index(idx)
        
        # Skip if this is the last row in the group
        if current_pos >= len(group_idx_list) - 1:
            continue
        
        # Get future indices (much faster than searching)
        future_indices = group_idx_list[current_pos + 1:]
        
        # Extract future quantities using numpy indexing
        future_q1 = q1_arr[future_indices].copy()
        future_q2 = q2_arr[future_indices].copy()
        
        # Calculate weighted averages for each bin
        bin_averages = {}
        remaining_q1 = future_q1.copy()
        remaining_q2 = future_q2.copy()
        
        for bin_num in range(max_bins):
            bin_start = bin_num * bin_size
            bin_end = (bin_num + 1) * bin_size
            
            # Track cumulative quantity1 and weighted sum
            cumsum_q1 = 0
            weighted_sum = 0
            events_to_remove = []
            
            for i, (q1, q2) in enumerate(zip(remaining_q1, remaining_q2)):
                # Check if this event completes or exceeds the bin
                if cumsum_q1 + q1 >= bin_end:
                    # This event crosses the bin boundary
                    needed_q1 = bin_end - cumsum_q1
                    
                    if needed_q1 > 0:
                        # Add partial contribution from this event
                        weighted_sum += needed_q1 * q2
                        cumsum_q1 += needed_q1
                        
                        # Update remaining quantity for this event
                        remaining_q1[i] = q1 - needed_q1
                    
                    break
                else:
                    # This event fits entirely within the current bin
                    weighted_sum += q1 * q2
                    cumsum_q1 += q1
                    events_to_remove.append(i)
            
            # Calculate weighted average for this bin
            if cumsum_q1 > bin_start:
                actual_weight = cumsum_q1 - bin_start
                if actual_weight > 0:
                    bin_averages[f'bin_{bin_num+1}_avg'] = weighted_sum / actual_weight
                else:
                    bin_averages[f'bin_{bin_num+1}_avg'] = np.nan
            else:
                bin_averages[f'bin_{bin_num+1}_avg'] = np.nan
            
            # Remove fully used events
            if events_to_remove:
                remaining_q1 = np.delete(remaining_q1, events_to_remove)
                remaining_q2 = np.delete(remaining_q2, events_to_remove)
            
            # If no more events, fill remaining bins with NaN
            if len(remaining_q1) == 0:
                for future_bin in range(bin_num + 1, max_bins):
                    bin_averages[f'bin_{future_bin+1}_avg'] = np.nan
                break
        
        # Create result row with original column names
        result_row = {
            id1_col: current_row[id1_col],
            id2_col: current_row[id2_col],
            timestamp_col: current_row[timestamp_col],
            q1_col: current_row[q1_col],
            q2_col: current_row[q2_col]
        }
        result_row.update(bin_averages)
        results.append(result_row)
    
    # Convert results to DataFrame
    result_df = pd.DataFrame(results)
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Weighted bin calculation completed in {elapsed_time:.4f} seconds")
    
    return result_df


def demonstrate_usage():
    """Demonstrate how to use the function with different column names."""
    
    print("="*80)
    print("DEMONSTRATION: Weighted Bin Calculation")
    print("="*80)
    
    # Create sample data with different column names
    sample_data = pd.DataFrame({
        'Asset': ['A', 'A', 'A', 'A', 'A'],
        'Type': ['x', 'x', 'x', 'x', 'x'],
        'Time': pd.date_range('2024-01-01', periods=5, freq='h'),
        'Volume': [50, 75, 100, 25, 150],
        'Price': [10, 20, 30, 40, 50]
    })
    
    print("Sample data with custom column names:")
    print(sample_data)
    print("\n" + "="*80)
    
    # Calculate weighted bins with custom column names
    result = calculate_weighted_bins(
        df=sample_data,
        id1_col='Asset',
        id2_col='Type', 
        timestamp_col='Time',
        q1_col='Volume',
        q2_col='Price',
        bin_size=100,
        max_bins=3
    )
    
    print("Results:")
    print(result)
    print("\n" + "="*80)
    
    # Explain the calculation
    print("Explanation for first row:")
    print("Future events: rows 1-4")
    print("Bin 1 (0-100):")
    print("  - Event 1: Volume=75, Price=20 → contributes 75")
    print("  - Event 2: Volume=100, Price=30 → need only 25 to reach 100")
    print("  - Weighted average = (75*20 + 25*30) / 100 = 22.5")
    print("Bin 2 (100-200):")
    print("  - Event 2 remainder: Volume=75, Price=30 → contributes 75")
    print("  - Event 3: Volume=25, Price=40 → contributes 25")
    print("  - Weighted average = (75*30 + 25*40) / 100 = 32.5")
    print("Bin 3 (200-300):")
    print("  - Event 4: Volume=150, Price=50 → need only 100 to reach 300")
    print("  - Weighted average = (100*50) / 100 = 50.0")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_usage()
    
    # Example of loading and processing a CSV file
    print("\n" + "="*80)
    print("EXAMPLE: Processing a CSV file")
    print("="*80)
    
    try:
        # Try to load the sample data
        df = pd.read_csv('sample_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"Loaded {len(df):,} rows from sample_data.csv")
        print("Processing with default column names...")
        
        result = calculate_weighted_bins(df, bin_size=100, max_bins=5)
        
        print(f"Results shape: {result.shape}")
        print("First few rows:")
        print(result.head())
        
        # Save results
        result.to_csv('weighted_bin_results.csv', index=False)
        print("Results saved to 'weighted_bin_results.csv'")
        
    except FileNotFoundError:
        print("sample_data.csv not found. Run generate_data.py first to create sample data.")
    except Exception as e:
        print(f"Error processing file: {e}") 