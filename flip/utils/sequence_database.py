"""
This script defines a SequenceDatabase class for storing protein sequences, their targets, and annotations.
It also provides functionality to encode sequences using different methods and to convert a CSV file 
into a SequenceDatabase object.
"""

from typing import List, Optional, Union
import pandas as pd
from .sequence_encoding import sequence_to_blosum62, sequence_to_one_hot, sequence_to_esm2

class SequenceDatabase:
    """
    A class to represent a database of protein sequences, their targets, and annotations.

    Attributes:
        sequences (list): A list to store encoded sequences.
        targets (list): A list to store target values (e.g., fitness scores) for each sequence.
        sets (list): A list to store the set type for each sequence (e.g., train, test).
        validations (list): A list to store the validation values for each sequence.
    """
    def __init__(self):
        """
        Initializes the SequenceDatabase with empty lists for sequences, targets, sets, and validations.
        """
        self.sequences = []
        self.targets = []
        self.sets = []
        self.validations = []

    def add_sequence(self, sequence: str, target: Optional[Union[int, float, str]] = None, set_type: str = 'train', validation: Optional[str] = None, encoding: str = 'blosum62'):
        """
        Adds a sequence to the database with its target, set, and validation, and encodes the sequence based on the specified encoding type.

        Args:
            sequence (str): The amino acid sequence to be added.
            target (Optional[Union[int, float, str]]): The target value associated with the sequence (e.g., fitness score).
            set_type (str): The set type for the sequence (e.g., train, test).
            validation (Optional[str]): The validation value for the sequence.
            encoding (str): The type of encoding to use for the sequence ('blosum62', 'one_hot', or 'esm2').

        Raises:
            ValueError: If an unknown encoding type is specified.
        """
        if encoding == 'blosum62':
            encoded_sequence = sequence_to_blosum62(sequence)
        elif encoding == 'one_hot':
            encoded_sequence = sequence_to_one_hot(sequence)
        elif encoding == 'esm2':
            encoded_sequence = sequence_to_esm2(sequence)
        else:
            raise ValueError(f"Unknown encoding type: {encoding}")
        
        self.sequences.append(encoded_sequence)
        self.targets.append(target)
        self.sets.append(set_type)
        self.validations.append(validation)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the database to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the sequences, targets, sets, and validations.
        """
        data = {
            'sequence': self.sequences,
            'target': self.targets,
            'set': self.sets,
            'validation': self.validations
        }
        return pd.DataFrame(data)

def read_csv_to_sequencedatabase(csv_path: str, encoding: str = 'blosum62') -> SequenceDatabase:
    """
    Reads a CSV file and converts it into a SequenceDatabase object.

    Args:
        csv_path (str): The path to the CSV file.
        encoding (str): The type of encoding to use for the sequences ('blosum62', 'one_hot', or 'esm2').

    Returns:
        SequenceDatabase: An object containing sequences, targets, sets, and validations from the CSV file.
    """
    df = pd.read_csv(csv_path)

    sequences = df['sequence'].tolist()
    targets = df['target'].tolist()
    sets = df['set'].tolist()
    validations = df['validation'].tolist()

    sequence_database = SequenceDatabase()

    for seq, target, set_type, validation in zip(sequences, targets, sets, validations):
        sequence_database.add_sequence(sequence=seq, target=target, set_type=set_type, validation=validation, encoding=encoding)
    
    return sequence_database

