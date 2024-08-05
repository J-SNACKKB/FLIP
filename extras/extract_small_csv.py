"""
This script generates a smaller CSV file for testing purposes by processing a dataset of protein sequences.

Steps:
1. Extracts 'low_vs_high.csv' from 'splits.zip'.
2. Reads the CSV into a DataFrame.
3. Extracts the first 10 rows.
4. Modifies sequences by randomly deleting positions to create different lengths.
5. Saves the modified subset to 'examples/data/four_mutations_random_lengths.csv'.

Function:
    random_delete(sequence: str, max_deletions: int = 10) -> str:
        Randomly deletes positions from a sequence.

Usage:
    Run from the 'extras' directory:
        $ python3 extract_small_csv.py
"""

import pandas as pd
import zipfile

# Paths to the zip file and the CSV file within the zip archive
zip_path = '../splits/gb1/splits.zip'
csv_file_name = 'splits/low_vs_high.csv'  # Updated to reflect the nested path inside the zip

# Verify the contents of the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    print("Contents of the zip file:", zip_ref.namelist())

# Extract the specific CSV file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    if csv_file_name in zip_ref.namelist():
        zip_ref.extract(csv_file_name, '../splits/gb1/splits')
        print(f"Extracted {csv_file_name} from the zip file.")
    else:
        raise KeyError(f"There is no item named '{csv_file_name}' in the archive")

# Paths to the original and small CSV files
original_csv_path = '../splits/gb1/splits/splits/low_vs_high.csv'  # Path where it will be extracted
small_csv_path = '../examples/data/four_mutations_small.csv'

# Load the original CSV file
df = pd.read_csv(original_csv_path)

# Extract the first 10 rows
small_df = df.head(10)

# Save the new smaller CSV file
small_df.to_csv(small_csv_path, index=False)

print(f"Extracted the first 10 rows and saved to {small_csv_path}")