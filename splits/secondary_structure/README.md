### Dataset description

The secondary structure split stem from three different publications, cited at the end, which aims at predicting the conservation score of the residues of a protein sequence.

This is a well-known dataset and it is used to validate the behavior of code and models. Only provided a `sampled` split for this purpose.

### Full dataset

The full dataset is divided in `train.jsonl`, `val.jsonl` and `new_pisces.jsonl`, JSONL files with proteins for training, validation and testing. Each line is a sample with id, sequence, label and resolved. There are 9712 proteins for training, 1080 proteins for validation and 648 proteins for testing.

Due to the size of these files, they can be found at http://data.bioembeddings.com/public/FLIP/secondary_structure/.

### Splits

All splits are classification splits as explained prior. Train/Test splits are done as follows.

Splits ([semaphore legend](../../README.md#split-semaphore)):
- 🟢 `sampled`: Randomly split sequences into `train`/`test` with 95/5% probability.

All splits are contained in the `splits.zip` file. There is one `sequences.fasta` file with all the sequences of the splits in FASTA format and one FASTA file with the labels for each split, in this case only one `sampled.fasta`file.

The labels files are organized by sequence ID. Each sequence label has `SET` atribute (either `train` or `test`) and `VALIDATION` attribute (when True, these are sequences for train that may be used for validation (e.g. early stopping)). Example:
```
>Seq1 SET=train VALIDATION=False
HHHHHEEE
```

The labels are string encodings of sequences. For each amino acid there is a secondary structure class (`H` for helix, `E` for sheet, `C` for coil).

### Cite
From the publishers as Bibtex:
```
@Article{klausen2019netsurfp,
title={NetSurfP-2.0: Improved prediction of protein structural features by integrated deep learning},
author={Klausen, Michael Schantz and Jespersen, Martin Closter and Nielsen, Henrik and Jensen, Kamilla Kjaergaard and Jurtz, Vanessa Isabell and Soenderby, Casper Kaae and Sommer, Morten Otto Alexander and Winther, Ole and Nielsen, Morten and Petersen, Bent and others},
journal={Proteins: Structure, Function, and Bioinformatics},
volume={87},
number={6},
pages={520--527},
year={2019},
publisher={Wiley Online Library}
}
```
```
@Article{9477085,
author={Elnaggar, Ahmed and Heinzinger, Michael and Dallago, Christian and Rehawi, Ghalia and Wang, Yu and Jones, Llion and Gibbs, Tom and Feher, Tamas and Angerer, Christoph and Steinegger, Martin and Bhowmik, Debsindhu and Rost, Burkhard},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
title={ProtTrans: Towards Cracking the Language of Lifes Code Through Self-Supervised Deep Learning and High Performance Computing},
year={2021},
volume={},
number={},
pages={1-1},
doi={10.1109/TPAMI.2021.3095381}
```
```
@Article{moult2018critical,
title={Critical assessment of methods of protein structure prediction (CASP)—Round XII},
author={Moult, John and Fidelis, Krzysztof and Kryshtafovych, Andriy and Schwede, Torsten and Tramontano, Anna},
journal={Proteins: Structure, Function, and Bioinformatics},
volume={86},
pages={7--15},
year={2018},
publisher={Wiley Online Library}
```

### Data licensing

RAW data was provided by the paper authors and is subject to [AFL-3](https://opensource.org/licenses/AFL-3.0).
Modified data available in this repository and in the `splits` falls under [AFL-3](https://opensource.org/licenses/AFL-3.0).

This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/), which permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.