### Dataset description

The six Bind splits stem from a [??? publication](???) which aims at predicting neutral/effect Single Aminoacid Variants (SAVs) in proteins.

### Full dataset

The full dataset is divided in one `sequences.fasta` file with all the wildtypes and different `.effect` files with the effect and neutral variants. The training/test folds are 9. Number 0 to 8 for testing and number 9 for testing.

Due to the size of these files, they can be found at http://data.bioembeddings.com/public/FLIP/savs/.

### Splits

All splits are classification splits as explained prior. Train/Test splits are done as follows.

Splits ([semaphore legend](../../README.md#split-semaphore)):
- `mixed`: train on wildtypes and training folds 0-8, test on test fold 9
- `human`: same as mixed but only with human proteins
- `only_savs`: same as mixed but without wildtypes

All splits are contained in the `splits.zip` file. These are CSV with colums:

- `sequence`: the AA sequence. May contain special characters!
- `set`: either `train` or `test`, if the sequence should be used for training or testing your model!
- `validation`: When True, these are sequences for train that may be used for validation (e.g. early stopping).
- `target`: the prediction target, which is a string encoding of the sequence. Each AA is encoded as 0 for not mutated, 1 for neutral SAV and 2 for effect SAV.

### Cite
From the publishers as Bibtex:


### Data licensing

RAW data downloaded from the two aforementioned publications is not subject to license.
Modified data available in this repository and in the `splits` falls under [AFL-3](https://opensource.org/licenses/AFL-3.0).

This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/), which permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.