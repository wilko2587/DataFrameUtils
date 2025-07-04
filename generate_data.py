import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(n_rows=1000, seed=42):
    """
    Generate a pandas DataFrame with sample data for testing.
    
    Parameters:
    -----------
    n_rows : int
        Number of rows to generate (default: 1000)
    seed : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: ID1, ID2, timestamp, quantity1, quantity2
    """
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate ID1 values (5 different values)
    id1_values = [f'ID{i:03d}' for i in range(1, 6)]
    id1 = np.random.choice(id1_values, size=n_rows)
    
    # Generate ID2 values ('a' or 'b' with 50/50 chance)
    id2 = np.random.choice(['a', 'b'], size=n_rows)
    
    # Generate timestamps spanning 30 days with multiple per day
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=30)
    
    # Generate random timestamps
    timestamps = []
    for _ in range(n_rows):
        random_date = start_date + timedelta(
            days=np.random.randint(0, 30),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60)
        )
        timestamps.append(random_date)
    
    # Sort timestamps
    timestamps.sort()
    
    # Generate strictly positive random values for quantities
    quantity1 = np.random.uniform(0, 1000, n_rows)
    quantity2 = np.random.uniform(0, 1000, n_rows)
    
    # Create DataFrame
    df = pd.DataFrame({
        'ID1': id1,
        'ID2': id2,
        'timestamp': timestamps,
        'quantity1': quantity1,
        'quantity2': quantity2
    })
    
    return df

def generate_large_sample_data(n_rows=1825000, seed=42):
    """
    Generate a large pandas DataFrame for performance testing.
    
    Parameters:
    -----------
    n_rows : int
        Number of rows to generate (default: 1,825,000)
    seed : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    pandas.DataFrame
        Large DataFrame with columns: ID1, ID2, timestamp, quantity1, quantity2
    """
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate ID1 values (5000 different values)
    id1_values = [f'ID{i:06d}' for i in range(1, 5001)]
    id1 = np.random.choice(id1_values, size=n_rows)
    
    # Generate ID2 values ('a' or 'b' with 50/50 chance)
    id2 = np.random.choice(['a', 'b'], size=n_rows)
    
    # Generate timestamps spanning 1 year with ~5000 events per day
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=365)
    
    # Generate random timestamps
    timestamps = []
    for _ in range(n_rows):
        random_date = start_date + timedelta(
            days=np.random.randint(0, 365),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60)
        )
        timestamps.append(random_date)
    
    # Sort timestamps
    timestamps.sort()
    
    # Generate strictly positive random values for quantities
    quantity1 = np.random.uniform(0, 1000, n_rows)
    quantity2 = np.random.uniform(0, 1000, n_rows)
    
    # Create DataFrame
    df = pd.DataFrame({
        'ID1': id1,
        'ID2': id2,
        'timestamp': timestamps,
        'quantity1': quantity1,
        'quantity2': quantity2
    })
    
    return df

def main():
    """Generate and save sample data."""
    
    print("Generating sample data...")
    
    # Generate small sample data
    df_small = generate_sample_data(n_rows=1000, seed=42)
    df_small.to_csv('sample_data.csv', index=False)
    print(f"Small sample data saved: {len(df_small):,} rows")
    
    # Generate large sample data
    print("Generating large sample data (this may take a moment)...")
    df_large = generate_large_sample_data(n_rows=1825000, seed=42)
    df_large.to_csv('large_sample_data.csv', index=False)
    print(f"Large sample data saved: {len(df_large):,} rows")
    
    # Print summary statistics
    print("\nData Summary:")
    print(f"Small dataset: {len(df_small):,} rows")
    print(f"Large dataset: {len(df_large):,} rows")
    print(f"Date range: {df_small['timestamp'].min()} to {df_small['timestamp'].max()}")
    print(f"Unique ID1 values: {len(df_small['ID1'].unique())}")
    print(f"Unique ID2 values: {sorted(df_small['ID2'].unique())}")
    
    print("\nSample data preview:")
    print(df_small.head())
    
    print("\nData generation completed successfully!")

if __name__ == "__main__":
    main()