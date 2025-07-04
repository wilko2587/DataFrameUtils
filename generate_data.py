import pandas as pd
import numpy as np

def generate_sample_data(n_rows=1000, n_id1s=5, n_id2s=3, save_to_csv=True):
    """
    Generate sample data for testing the weighted bin calculator.
    
    Parameters:
    -----------
    n_rows : int
        Number of rows to generate (default: 1000)
    n_id1s : int
        Number of unique ID1 values (default: 5)
    n_id2s : int
        Number of unique ID2 values (default: 3)
    save_to_csv : bool
        Whether to save the data to CSV (default: True)
    
    Returns:
    --------
    pandas.DataFrame
        Generated sample data
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate ID1 and ID2 values
    id1s = [f'ID1_{i}' for i in range(n_id1s)]
    id2s = [f'ID2_{i}' for i in range(n_id2s)]
    
    # Create random data
    data = {
        'ID1': np.random.choice(id1s, n_rows),
        'ID2': np.random.choice(id2s, n_rows),
        'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='h'),
        'quantity1': np.random.randint(10, 201, n_rows),  # Volume-like values
        'quantity2': np.random.randint(5, 51, n_rows)     # Price-like values
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by ID1, ID2, and timestamp to ensure chronological order within groups
    df = df.sort_values(['ID1', 'ID2', 'timestamp']).reset_index(drop=True)
    
    if save_to_csv:
        df.to_csv('sample_data.csv', index=False)
        print(f"Generated {n_rows:,} rows of sample data")
        print(f"Saved to 'sample_data.csv'")
        print(f"ID1 values: {id1s}")
        print(f"ID2 values: {id2s}")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Quantity1 range: {df['quantity1'].min()} to {df['quantity1'].max()}")
        print(f"Quantity2 range: {df['quantity2'].min()} to {df['quantity2'].max()}")
    
    return df


def generate_small_test_data():
    """Generate a small test dataset for quick testing."""
    return generate_sample_data(n_rows=100, n_id1s=3, n_id2s=2, save_to_csv=False)


if __name__ == "__main__":
    print("="*80)
    print("GENERATING SAMPLE DATA")
    print("="*80)
    
    # Generate sample data
    df = generate_sample_data()
    
    print("\n" + "="*80)
    print("SAMPLE DATA PREVIEW")
    print("="*80)
    print("First 10 rows:")
    print(df.head(10))
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    print("\n" + "="*80)
    print("DATA GENERATION COMPLETED!")
    print("="*80)