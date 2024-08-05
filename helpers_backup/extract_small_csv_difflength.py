import pandas as pd
import random

original_csv_path = '../splits/gb1/four_mutations_full_data.csv'
small_csv_path = 'four_mutations_random_lengths.csv'

# Loading the original CSV file
df = pd.read_csv(original_csv_path)

# Extracting the first 10 rows
small_df = df.head(10)

# Function to randomly delete positions from a sequence
# to create sequences of differeny lengths for testing 
# purposes 
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
