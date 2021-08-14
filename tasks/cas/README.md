### Dataset description

The original data was collected from the [supplementary material](https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-017-17081-y/MediaObjects/41598_2017_17081_MOESM2_ESM.xlsx) of [Spencer and Zhang](https://www.nature.com/articles/s41598-017-17081-y).
It represents a DMS set for Streptococcus pyogenes Cas9. While sequences for Cas9 are available from UniProt, e.g. [Q99ZW2](https://www.uniprot.org/uniprot/Q99ZW2), it appears that the wild type used by the authors of this study differs slightly from any sequence deposited in UniProt.
As such, the wildtype sequence is considered as the WT reported by the study, and not one deposited in databases like UniProt.

The selected measures from this manuscript were:
- Log2 Fold Change after Negative Selection
- Log2 Fold Change after Positive Selection


### Full dataset

The `positive_mutation_matrix.csv` and `negative_mutation_matrix.csv` are LxN matrices representing either the positive or negative selection score for each AA subsitution along the protein length (L).

### Tasks

Both tasks are regression tasks, one being the Log2 Fold change after positive selection, the other after negative selection. 
Train/Test splits are done on the [PI domain](https://pfam.xfam.org/family/PF16595), meaning: train sequences are those with mutations NOT in the PI domain, while test sequences are those with mutations in the PI domain.


All tasks are contained in the `tasks.zip` file. These are CSV with colums:

- `sequence`: the AA sequence. May contain special characters!
- `target`: the prediction target. This may be continuous (`regression`), or True/False (`binary`)
- `set`: either `train` or `test`, if the sequence should be used for training or testing your model!

### Cite
From the publisher:
> Spencer, J.M., Zhang, X. Deep mutational scanning of S. pyogenes Cas9 reveals important functional domains. Sci Rep 7, 16836 (2017). https://doi.org/10.1038/s41598-017-17081-y


Bibtex:
```
@article{spencer2017deep,
  title={Deep mutational scanning of S. pyogenes Cas9 reveals important functional domains},
  author={Spencer, Jeffrey M and Zhang, Xiaoliu},
  journal={Scientific reports},
  volume={7},
  number={1},
  pages={1--14},
  year={2017},
  publisher={Nature Publishing Group}
}
```

### Data licensing

The RWA data downloaded from Springer Nature is subject to [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Modified data available in this repository and in the `tasks` falls under [AFL-3](https://opensource.org/licenses/AFL-3.0).