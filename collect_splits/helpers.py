import matplotlib.pyplot as plt

from typing import List

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def read_fasta(path: str) -> List[SeqRecord]:
    """
    Helper function to read FASTA file.
    :param path: path to a valid FASTA file
    :return: a list of SeqRecord objects.
    """
    try:
        return list(SeqIO.parse(path, "fasta"))
    except FileNotFoundError:
        raise  # Already says "No such file or directory"
    except Exception as e:
        raise ValueError(f"Could not parse '{path}'. Are you sure this is a valid fasta file?") from e
    pass


def plot_data_statistics(dataframe, column_name, fitness_colum):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    dataframe[f'{column_name}'].hist(ax=ax[0,0])
    dataframe.query(f'{column_name}=="train"')[fitness_colum].hist(ax=ax[1,0])
    dataframe.query(f'{column_name}=="test"')[fitness_colum].hist(ax=ax[1,1])
    ax[0,1].set_visible(False)

    test_count = len(dataframe.query(f'{column_name}=="test"'))
    total = len(dataframe.query(f'{column_name}=="train"')) + test_count
    test_percentage = (test_count*100)/total
    
    print(f'The test set will be {test_percentage:.2f}% of the data ({test_count} out of {total} samples).')

    
# Appliccable only to AAV
def print_positive_stats(dataframe, column_name):
    
    print(f"""Set {column_name} has {
      len(dataframe.query(f"{column_name}=='train'"))
    } train and {
      len(dataframe.query(f"{column_name}=='test'"))
    } test sequences. {
      round(len(dataframe.query(f"{column_name}=='test' & binary_score==True")) /
      len(dataframe.query(f"{column_name}=='test'")) * 100)
    }% of test is positive.""")
    
