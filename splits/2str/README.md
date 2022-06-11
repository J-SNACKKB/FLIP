### Dataset description

The 2STR split stem from a [???? publication](????) which aims at predicting the conservation score of the residues of a protein sequence.

This is a well-known dataset and it is used to validate the behavior of code and models. Only provided a `sampled` split for this purpose.

### Full dataset

The full dataset is divided in `train.jsonl`, a JSONL file with proteins for training. Each line is a sample with id, sequence, label and resolved (ignored) (9712 proteins); `val.jsonl`, a JSONL file with proteins for validation. Each line is a sample with id, sequence, label and resolved (ignored) (1080 proteins); and `new_pisces.jsonl`, a JSONL file with proteins for testing. Each line is a sample with id, sequence, label and resolved (ignored) (648 proteins).

Due to the size of these files, they can be found at http://data.bioembeddings.com/public/FLIP/2str/.

### Splits

All splits are classification splits as explained prior. Train/Test splits are done as follows.

Splits ([semaphore legend](../../README.md#split-semaphore)):
- 🟠 `sampled`: Randomly split sequences into `train`/`test` with 95/5% probability.

All splits are contained in the `splits.zip` file. These are CSV with colums:

- `sequence`: the AA sequence. May contain special characters!
- `set`: either `train` or `test`, if the sequence should be used for training or testing your model!
- `validation`: When True, these are sequences for train that may be used for validation (e.g. early stopping).
- `target`: the prediction target, which is string encoding of the sequence. For each amino acid there is a secondary structure class (`H` for helix, `E` for sheet, `C` for coil).

### Cite
From the publishers as Bibtex:
```
???
```

### Data licensing

RAW data was provided by the paper authors and is not subject to license.
Modified data available in this repository and in the `splits` falls under [AFL-3](https://opensource.org/licenses/AFL-3.0).

This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/), which permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.