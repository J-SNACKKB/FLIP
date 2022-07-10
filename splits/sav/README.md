### Dataset description

The three SAV splits stem from a [2021 publication](https://link.springer.com/article/10.1007/s00439-021-02411-y) and a [2015 publication](https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-16-S8-S1) which aim at predicting neutral/effect Single Aminoacid Variants (SAVs) in proteins.

### Full dataset

The full dataset is divided in one `sequences.fasta` file with all the wildtypes and different `.effect` files with the effect and neutral variants. The training/test folds are 9. Number 0 to 8 for testing and number 9 for testing.

Due to the size of these files, they can be found at http://data.bioembeddings.com/public/FLIP/savs/.

### Splits

All splits are classification splits as explained prior. Train/Test splits are done as follows.

Splits ([semaphore legend](../../README.md#split-semaphore)):
- `mixed`: train on wildtypes (with target `neutral`) and training folds 0-8, test on test fold 9
- `human`: same as mixed but only with human proteins
- `only_savs`: same as mixed but without wildtypes

All splits are contained in the `splits.zip` file. These are CSV with colums:

- `sequence`: the AA sequence. May contain special characters!
- `set`: either `train` or `test`, if the sequence should be used for training or testing your model!
- `validation`: When True, these are sequences for train that may be used for validation (e.g. early stopping).
- `target`: the prediction target, which can be `effect` or `neutral`

### Cite
From the publishers as Bibtex:
```
@article{marquet2021embeddings,
  title={Embeddings from protein language models predict conservation and variant effects},
  author={Marquet, C{\'e}line and Heinzinger, Michael and Olenyi, Tobias and Dallago, Christian and Erckert, Kyra and Bernhofer, Michael and Nechaev, Dmitrii and Rost, Burkhard},
  journal={Human genetics},
  pages={1--19},
  year={2021},
  publisher={Springer}
}
```
```
@article{hecht2015better,
  title={Better prediction of functional effects for sequence variants},
  author={Hecht, Maximilian and Bromberg, Yana and Rost, Burkhard},
  journal={BMC genomics},
  volume={16},
  number={8},
  pages={1--12},
  year={2015},
  publisher={BioMed Central}
}
```

### Data licensing

RAW data downloaded from the two aforementioned publications is not subject to license.
Modified data available in this repository and in the `splits` falls under [AFL-3](https://opensource.org/licenses/AFL-3.0).

This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/), which permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.