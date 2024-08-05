import pandas as pd
import random
import zipfile

zip_file_path = '../splits/gb1/splits.zip'
csv_file_name = 'splits/low_vs_high.csv'
small_csv_path = '../examples/data/four_mutations_random_lengths.csv'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    if csv_file_name in zip_ref.namelist():
        zip_ref.extract(csv_file_name, '../splits/gb1/')
    else:
        raise KeyError(f"There is no item named '{csv_file_name}' in the archive")

extracted_csv_path = '../splits/gb1/' + csv_file_name
df = pd.read_csv(extracted_csv_path)

# Extracting the first 10 rows
small_df = df.head(10)

# Function to randomly delete positions from a sequence
# to create sequences of different lengths for testing
def random_delete(sequence, max_deletions=10):
    num_deletions = random.randint(1, max_deletions)
    sequence_list = list(sequence)
    for _ in range(num_deletions):
        if sequence_list:
            del sequence_list[random.randint(0, len(sequence_list) - 1)]
    return ''.join(sequence_list)

small_df['sequence'] = small_df['sequence'].apply(random_delete)

# Saving the new smaller CSV file
small_df.to_csv(small_csv_path, index=False)

print(f"Extracted the first 10 rows, modified sequences, and saved to {small_csv_path}")
