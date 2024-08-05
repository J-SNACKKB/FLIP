"""
This script provides functions for encoding protein sequences using different methods.
It includes BLOSUM62, one-hot encoding, and a placeholder for ESM2 encoding.

Functions:
    sequence_to_blosum62: Converts a protein sequence to a BLOSUM62 encoded matrix.
    sequence_to_one_hot: Converts a protein sequence to a one-hot encoded matrix.
    sequence_to_esm2: Placeholder function for converting a protein sequence to an ESM2 encoded tensor.
"""

import numpy as np
from Bio.Align import substitution_matrices
from ..zero_shot.global_parameters import ALL_AAS

def sequence_to_blosum62(sequence: str) -> np.ndarray:
    """
    Converts a protein sequence to a BLOSUM62 encoded matrix.
    
    Parameters:
        sequence (str): Protein sequence.
        
    Returns:
        np.ndarray: BLOSUM62 encoded matrix.
    """
    blosum62 = substitution_matrices.load("BLOSUM62")
    length = len(sequence)
    encoded_matrix = np.zeros((length, 20))

    amino_acids = ALL_AAS
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

    for i, aa in enumerate(sequence):
        if aa in aa_to_index:
            encoded_matrix[i] = [blosum62.get((aa, b), 0) for b in amino_acids]
        else:
            encoded_matrix[i] = [0] * 20
    
    return encoded_matrix

def sequence_to_one_hot(sequence: str) -> np.ndarray:
    """
    Converts a protein sequence to a one-hot encoded matrix.
    
    Parameters:
        sequence (str): Protein sequence.
        
    Returns:
        np.ndarray: One-hot encoded matrix.
    """
    amino_acids = ALL_AAS
    encoding_dict = {aa: i for i, aa in enumerate(amino_acids)}
    length = len(sequence)
    one_hot_matrix = np.zeros((length, len(amino_acids)))

    for i, aa in enumerate(sequence):
        if aa in encoding_dict:
            one_hot_matrix[i, encoding_dict[aa]] = 1
    
    return one_hot_matrix

def sequence_to_esm2(sequence: str) -> np.ndarray:
    """
    Placeholder function for converting a protein sequence to an ESM2 encoded tensor.
    
    Parameters:
        sequence (str): Protein sequence.
        
    Returns:
        np.ndarray: ESM2 encoded tensor (placeholder).
    """
    # ESM2 encoding logic will be added later
    return np.random.rand(len(sequence), 1280)
