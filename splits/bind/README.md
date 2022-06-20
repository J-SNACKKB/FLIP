### Dataset description

The six Bind splits stem from a [2021 publication](https://www.nature.com/articles/s41598-021-03431-4) which aims at predicting binding of residues to small, metal or nuclear molecules.

### Full dataset

The full dataset is divided in a development set and an independent set.The development set contains `all.fasta` with all information associated to proteins; `binding_residues_2.5_metal.txt`, `binding_residues_2.5_nuclear.txt` and `binding_residues_2.5_small.txt` with the binding residues for each protein ID; and `uniprot_test.txt` contains the IDs of the proteins for testing (set named TestSet300). The independent set contains the same files but with proteins originally selected only for testing (set named TestSetNew46).

Due to the size of these files, they can be found at http://data.bioembeddings.com/public/FLIP/bind/.

### Legend 

As mentioned, the datasets work with 3 types of bindign residues:
1. (S) Small molecules (S)
2. (N) Nucleic acids (DNA and RNA) 
3. (M) Metal ions (M)

To encode this information, we use the following labels, following the binary pattern (SNM):
1. 000 = No binding
2. 001 = Metal
3. 010 = Nucleic
4. 011 = Nucleic + Metal
5. 100 = Small
6. 101 = Small + Metal
7. 110 = Small + Nucleic
8. 111 = Small + Nucleic + Metal

This encoding allows for single-class multiclass and multi-class classification of the binding residues. The final encoding consists of integers from 0 to 7 for each residue, following the binary encoding.

### Splits

All splits are classification splits as explained prior. Train/Test splits are done as follows.

Splits ([semaphore legend](../../README.md#split-semaphore)):
- Working at protein-level (train on sequences with X ligan type(s), test on else (~18% of the proteins have >1 type of ligand)):
    - 🟢 `one_vs_many`: train on proteins with only 1 type of ligand, test on proteins with 2 and 3 types of ligands
    - 🟢 `two_vs_many`: train on proteins with 1 or 2 types of ligand, test on proteins with 3 types of ligand
    - 🟢 `three_vs_many`: train on proteins with 1, 2 or 3 types of ligand from original training set, test on original test sets TestSet300 and TestSetNew46 mixed.

- Working at residue-level (train on sequences with residues assigned to only 1 type of ligand, test on sequences with residues assigned to multiple classes (~4% of the residues have more than one type of ligand)):
    - 🟢 `one_vs_sm`: train on proteins with residues having only one type of ligand, test on proteins with residues having Small+Metal ligands
    - 🟢 `one_vs_mn`: train as `one_vs_sm` but with balances classes, test on proteins with residues having Metal+Nuclear
    - 🟢 `one_vs_sn`: train as `one_vs_sm` but with balances classes, test on proteins with residues having Small+Nuclear

All splits are contained in the `splits.zip` file. There are one `sequences.fasta` file with all the sequences of the splits in FASTA format and one FASTA file with the labels for each split, i.e. `one_vs_many.fasta`, `two_vs_many.fasta`, etc.

The labels files are organized by sequence ID. Each sequence label has `SET` atribute (either `train` or `test`) and `VALIDATION` attribute (when True, these are sequences for train that may be used for validation (e.g. early stopping)). Example:
```
>Seq1 SET=train VALIDATION=False
DVCDVVDD
```

The labels are string encodings of sequences. Following the pattern SNM (for Small Nuclear Metal): 000 = No biding, 001 = Metal, 010 = Nuclear, 100 = Small. If we have a multiclass binding residue, e.g. has both Small and Metal, its encoding is 101 (for Small+Metal). This results in a string with integers from 0 to 7, where 0 = no binding, 1 = Metal, 2 = Nuclar, 3 = Nuclear+Metal, 4 = Small, 5 = Small+Metal, 6 = Small+Nuclear, 7 = Small+Nuclear+Metal.

### Cite
From the publishers as Bibtex:
```
@article{littmann2021protein,
title={Protein embeddings and deep learning predict binding residues for various ligand classes},
author={Littmann, Maria and Heinzinger, Michael and Dallago, Christian and Weissenow, Konstantin and Rost, Burkhard},
journal={Scientific reports},
volume={11},
number={1},
pages={1--15},
year={2021},
abstract = "{One important aspect of protein function is the binding of proteins to ligands, including small molecules, metal ions, and macromolecules such as DNA or RNA. Despite decades of experimental progress many binding sites remain obscure. Here, we proposed bindEmbed21, a method predicting whether a protein residue binds to metal ions, nucleic acids, or small molecules. The Artificial Intelligence (AI)-based method exclusively uses embeddings from the Transformer-based protein Language Model (pLM) ProtT5 as input. Using only single sequences without creating multiple sequence alignments (MSAs), bindEmbed21DL outperformed MSA-based predictions. Combination with homology-based inference increased performance to F1 = 48 ± 3% (95% CI) and MCC = 0.46 ± 0.04 when merging all three ligand classes into one. All results were confirmed by three independent data sets. Focusing on very reliably predicted residues could complement experimental evidence: For the 25% most strongly predicted binding residues, at least 73% were correctly predicted even when ignoring the problem of missing experimental annotations. The new method bindEmbed21 is fast, simple, and broadly applicable—neither using structure nor MSAs. Thereby, it found binding residues in over 42% of all human proteins not otherwise implied in binding and predicted about 6% of all residues as binding to metal ions, nucleic acids, or small molecules.}",
publisher={Nature Publishing Group}
}
```

### Data licensing

RAW data downloaded from the two aforementioned publications is not subject to license.
Modified data available in this repository and in the `splits` falls under [AFL-3](https://opensource.org/licenses/AFL-3.0).

This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/), which permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.