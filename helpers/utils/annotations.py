
from enum import Enum
from typing import Optional, Union, List
from hashlib import md5
from pathlib import Path

from pandas import DataFrame
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


_AA = Enum('_AA', 
           ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 
            'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 
            'Y', 'V']
            )

TrainingSet = Enum('TrainingSet', ['train', 'test', 'val'])

class AA:
    resolved: bool = True
    aa: _AA

    def __init__(self, aa: chr) -> None:
        self.aa = _AA(aa)

class Annotation:
    name: str
    value: str

    def __init__(self, name:str, value:str) -> None:
        self.name = name
        self.value = value

class Sequence:
    target: Optional[Union[int, float, str]]
    training_set: Optional[TrainingSet] = None
    sequence: List[AA]
    id: str
    annotations: List[Annotation]

    def __init__(self) -> None:
        self.annotations = list()

    def add_sequence(self, seq: str):
        self.sequence = [AA(aa) for aa in seq]

    def get_sequence(self):
        return "".join([aa.aa for aa in self.sequence])


class SequenceDataset(list):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def append(self, sequence: str, set: TrainingSet, target: Union[int, float, str], 
               id: Optional[str] = None, **kwargs):
        s = Sequence()

        for key, val in kwargs:
            s.annotations.append(Annotation(name=key, value=val))

        if not id:
            s.id = str(md5(sequence.encode()).hexdigest())
        else:
            s.id = id
        
        s.target = target

        s.add_sequence(sequence)

        self.append(s)

    def write_as_fasta(self, out_path: Path):
            SeqIO.write([
                SeqRecord(
                    Seq(protein.get_sequence()),
                    id=f"{protein.id}",
                    description=f"SET={protein.training_set};"
                                f"TARGET={protein.target};"
                                f";".join([
                                    f"{str(annotation.name).upper()}={annotation.value}" for
                                      annotation in protein.annotations
                                      ])
                ) for protein in self
            ], out_path, "fasta")

    def to_dataframe(self):
        pass

    def to_dict(self):
        return {
            protein.id: {
                'sequence': protein.get_sequence(),
                'target': protein.target,
                'set': protein.training_set,
                'annotations': {
                    annotation.name: annotation.value for annotation in protein.annotations
                }
            } for protein in self
        }
