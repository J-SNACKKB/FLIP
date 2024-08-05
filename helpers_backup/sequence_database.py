"""
This script defines a SequenceDatabase class for storing protein sequences, their targets, and annotations.
It also provides functionality to encode sequences using different methods and to convert a CSV file 
into a SequenceDatabase object.
"""

from typing import List, Optional, Union
import pandas as pd
from sequence_encoding import sequence_to_blosum62, sequence_to_one_hot, sequence_to_esm2

class SequenceDatabase:
    """
    A class to represent a database of protein sequences, their targets, and annotations.

    Attributes:
        sequences (list): A list to store encoded sequences.
        targets (list): A list to store target values (e.g., fitness scores) for each sequence.
        annotations (list): A list to store annotations for each sequence.
    """
    def __init__(self):
        """
        Initializes the SequenceDatabase with empty lists for sequences, targets, and annotations.
        """
        self.sequences = []
        self.targets = []
        self.annotations = []

    def add_sequence(self, sequence: str, target: Optional[Union[int, float, str]] = None, annotations: Optional[dict] = None, encoding: str = 'blosum62'):
        """
        Adds a sequence to the database with its target and annotations, and encodes the sequence based on the specified encoding type.

        Args:
            sequence (str): The amino acid sequence to be added.
            target (Optional[Union[int, float, str]]): The target value associated with the sequence (e.g., fitness score).
            annotations (Optional[dict]): Additional annotations for the sequence. Expected keys include:
                - 'Variants': str, mutation variants.
                - 'HD': int, HD value.
                - 'Count input': int, initial count.
                - 'Count selected': int, selected count.
                - 'keep': bool, flag to keep the sequence.
                - 'one_vs_rest': str, training split for one vs rest.
                - 'one_vs_rest_validation': float or NaN, validation split for one vs rest.
                - 'two_vs_rest': str, training split for two vs rest.
                - 'two_vs_rest_validation': float or NaN, validation split for two vs rest.
                - 'three_vs_rest': str, training split for three vs rest.
                - 'three_vs_rest_validation': float or NaN, validation split for three vs rest.
                - 'sampled': str, training split for sampled data.
                - 'sampled_validation': float or NaN, validation split for sampled data.
                - 'low_vs_high': str, training split for low vs high.
                - 'low_vs_high_validation': float or NaN, validation split for low vs high.
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
        self.annotations.append(annotations)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the database to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the sequences, targets, and annotations.
        """
        data = {
            'sequence': self.sequences,
            'target': self.targets,
            'annotations': self.annotations
        }
        return pd.DataFrame(data)

def read_csv_to_sequencedatabase(csv_path: str, encoding: str = 'blosum62') -> SequenceDatabase:
    """
    Reads a CSV file and converts it into a SequenceDatabase object.

    Args:
        csv_path (str): The path to the CSV file.
        encoding (str): The type of encoding to use for the sequences ('blosum62', 'one_hot', or 'esm2').

    Returns:
        SequenceDatabase: An object containing sequences, targets, and annotations from the CSV file.
    """
    df = pd.read_csv(csv_path)

    sequences = df['sequence'].tolist()
    fitness_scores = df['Fitness'].tolist()

    additional_columns = [col for col in df.columns if col not in ['sequence', 'Fitness']]
    
    annotations_list = df[additional_columns].to_dict(orient='records')

    sequence_database = SequenceDatabase()

    for seq, fitness, annotations in zip(sequences, fitness_scores, annotations_list):
        sequence_database.add_sequence(sequence=seq, target=fitness, annotations=annotations, encoding=encoding)
    
    return sequence_database
