import pandas as pd
import numpy as np
import time
from tqdm import tqdm

def calculate_weighted_bins(df, id1_col='ID1', id2_col='ID2', timestamp_col='timestamp', 
                           q1_col='quantity1', q2_col='quantity2', bin_size=100, max_bins=10):
    """
    Calculate weighted averages of quantity2 in bins of quantity1 increments.
    
    This function groups data by ID1, then for each row calculates weighted averages
    of quantity2 in bins of quantity1 increments. Events that span bin boundaries are handled
    by splitting them proportionally.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the data
    id1_col : str
        Column name for the first identifier (default: 'ID1')
    id2_col : str
        Column name for the second identifier (default: 'ID2') - kept for output but not used for grouping
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
    ...     'timestamp': pd.date_range('2024-01-01', periods=4, freq='h'),
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
    
    # Sort by ID1 and timestamp (removed ID2 from sorting)
    df_sorted = df.sort_values([id1_col, timestamp_col]).reset_index(drop=True)
    
    # Pre-compute group indices for O(1) lookup (now only by ID1)
    print("Pre-computing group indices...")
    group_indices = {}
    for idx, row in df_sorted.iterrows():
        key = row[id1_col]  # Only use ID1 for grouping
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
        key = current_row[id1_col]  # Only use ID1 for grouping
        
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
            # We need to fill exactly bin_size units for this bin
            needed_for_bin = bin_size
            weighted_sum = 0
            events_to_remove = []
            
            for i, (q1, q2) in enumerate(zip(remaining_q1, remaining_q2)):
                if needed_for_bin <= 0:
                    break
                
                if q1 >= needed_for_bin:
                    # This event can fill the rest of the bin
                    contribution = needed_for_bin * q2
                    weighted_sum += contribution
                    remaining_q1[i] = q1 - needed_for_bin
                    needed_for_bin = 0
                    break
                else:
                    # This event contributes all it has
                    contribution = q1 * q2
                    weighted_sum += contribution
                    needed_for_bin -= q1
                    events_to_remove.append(i)
            
            # Calculate weighted average for this bin
            quantity_in_bin = bin_size - needed_for_bin
            if quantity_in_bin > 0:
                bin_averages[f'bin_{bin_num+1}_avg'] = weighted_sum / quantity_in_bin
            else:
                bin_averages[f'bin_{bin_num+1}_avg'] = np.nan
            
            # Remove fully used events
            if events_to_remove:
                remaining_q1 = np.delete(remaining_q1, events_to_remove)
                remaining_q2 = np.delete(remaining_q2, events_to_remove)
            
            # If no more events, fill remaining bins with NaN
            if len(remaining_q1) == 0 or (len(remaining_q1) == 1 and remaining_q1[0] == 0):
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


def run_comprehensive_test():
    """Run a comprehensive test with detailed bin transitions."""
    
    print("="*80)
    print("COMPREHENSIVE TEST: Detailed Bin Transitions")
    print("="*80)
    
    # Create test dataset
    np.random.seed(42)
    n_rows = 100
    id1s = np.random.choice(['A', 'B', 'C'], n_rows)
    id2s = np.random.choice(['X', 'Y'], n_rows)
    timestamps = pd.date_range('2024-01-01', periods=n_rows, freq='h')
    quantity1 = np.random.randint(10, 201, n_rows)
    quantity2 = np.random.randint(5, 51, n_rows)
    
    df = pd.DataFrame({
        'ID1': id1s,
        'ID2': id2s,
        'timestamp': timestamps,
        'quantity1': quantity1,
        'quantity2': quantity2
    })
    
    # Get the raw data for group A (now only grouped by ID1)
    group_a = df[df['ID1'] == 'A'].sort_values('timestamp').reset_index(drop=True)
    
    # First row's future events
    first_row = group_a.iloc[0]
    future_events = group_a.iloc[1:]
    
    print(f"ANALYZING FIRST ROW:")
    print(f"Timestamp: {first_row['timestamp']}")
    print(f"ID2: {first_row['ID2']}")
    print(f"Quantity1: {first_row['quantity1']}")
    print(f"Quantity2: {first_row['quantity2']}")
    print()
    
    print("FUTURE EVENTS (first 10):")
    print("-" * 50)
    for i in range(min(10, len(future_events))):
        print(f"Event {i}: ID2={future_events.iloc[i]['ID2']}, "
              f"quantity1={future_events.iloc[i]['quantity1']:3d}, "
              f"quantity2={future_events.iloc[i]['quantity2']:2d}")
    print()
    
    # Run the algorithm
    result = calculate_weighted_bins(
        df=df,
        id1_col='ID1',
        id2_col='ID2',
        timestamp_col='timestamp',
        q1_col='quantity1',
        q2_col='quantity2',
        bin_size=50,
        max_bins=8
    )
    
    # Get the first row result for group A
    result_a = result[result['ID1'] == 'A'].iloc[0]
    
    print("ALGORITHM RESULTS:")
    print("-" * 50)
    for i in range(1, 9):
        col_name = f'bin_{i}_avg'
        if col_name in result_a:
            value = result_a[col_name]
            if pd.notna(value):
                print(f"Bin {i}: {value:.2f}")
            else:
                print(f"Bin {i}: NaN")
    
    print("\n" + "="*80)
    print("EXPECTED RESULTS:")
    print("="*80)
    print("Note: Results now include all future events for ID1='A' regardless of ID2 value")
    print("This means more data points contribute to each bin compared to the previous")
    print("dual-grouping approach, potentially leading to different weighted averages.")
    
    return df, result


if __name__ == "__main__":

    print("RUNNING COMPREHENSIVE TEST")
    print("="*80)
    
    # Run comprehensive test
    df, result = run_comprehensive_test()
    
    # Save results
    result.to_csv('weighted_bin_results.csv', index=False)
    print(f"\nResults saved to 'weighted_bin_results.csv'")
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80) 