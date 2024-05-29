from hashlib import md5
from pathlib import Path
from typing import Optional

from pandas import read_csv

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


def csv_to_fasta(task_csv_path: Path, out_path: Optional[Path] = None):

    task_csv = read_csv(task_csv_path)
    
    if out_path == None:
        out_path = task_csv_path + ".fasta"

    protein_sequences = list()

    for i, protein in enumerate(task_csv.to_dict(orient='records')):
        protein_sequences.append(
            SeqRecord(
                Seq(protein.get('sequence')),
                id=f"S{i}",
                description=f"MD5={md5(protein.get('sequence').encode()).hexdigest()};"
                            f"SET={protein.get('set')};"
                            f"TARGET={protein.get('target')}"
            )
        )

    SeqIO.write(protein_sequences, out_path, "fasta")
