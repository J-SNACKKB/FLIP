import torch
from torch.utils.data import Dataset, DataLoader
from sequence_database import SequenceDatabase
from torch.nn.utils.rnn import pad_sequence

class ProteinDataset(Dataset):
    """
    A custom Dataset class for protein sequences.

    Attributes:
        encoded_sequences (list): A list of encoded protein sequences.
        targets (list): A list of target values corresponding to the sequences.

    Methods:
        __len__(): Returns the number of sequences in the dataset.
        __getitem__(idx): Returns the sequence and target at the specified index.
    """
    def __init__(self, sequence_database: SequenceDatabase):
        """
        Initializes the ProteinDataset with encoded sequences and targets from a SequenceDatabase.

        Args:
            sequence_database (SequenceDatabase): An object containing sequences and their corresponding targets.
        """
        self.encoded_sequences = sequence_database.sequences
        self.targets = sequence_database.targets

    def __len__(self):
        """
        Returns the number of sequences in the dataset.

        Returns:
            int: The number of sequences in the dataset.
        """
        return len(self.encoded_sequences)

    def __getitem__(self, idx):
        """
        Returns the sequence and target at the specified index.

        Args:
            idx (int): The index of the sequence and target to retrieve.

        Returns:
            tuple: A tuple containing the sequence tensor and the target tensor.
        """
        return torch.tensor(self.encoded_sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

def custom_collate_fn(batch):
    """
    A custom collate function to handle variable-length sequences in a batch.

    Args:
        batch (list): A list of tuples where each tuple contains a sequence tensor and a target tensor.

    Returns:
        tuple: A tuple containing padded sequences, sequence lengths, and targets.
    """
    sequences, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    targets = torch.tensor(targets)
    return padded_sequences, lengths, targets

def create_dataloader(dataset: ProteinDataset, batch_size: int):
    """
    Creates a DataLoader for the ProteinDataset.

    Args:
        dataset (ProteinDataset): The dataset to load.
        batch_size (int): The number of samples per batch.

    Returns:
        DataLoader: A DataLoader object for the dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
