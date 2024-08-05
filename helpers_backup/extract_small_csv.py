"""
This script generates a smaller CSV file for testing purposes
by extracting the first 10 rows from a larger CSV file.
"""

import pandas as pd

original_csv_path = '../splits/gb1/four_mutations_full_data.csv'
small_csv_path = 'four_mutations_small.csv'

# Load the original CSV file
df = pd.read_csv(original_csv_path)

# Extract the first 10 rows
small_df = df.head(10)

# Save the new smaller CSV file
small_df.to_csv(small_csv_path, index=False)

print(f"Extracted the first 10 rows and saved to {small_csv_path}")