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
