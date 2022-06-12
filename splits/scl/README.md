### Dataset description

The six SCL (SubCellularLocation) splits stems from a [2021 publication](https://academic.oup.com/bioinformaticsadvances/article/1/1/vbab035/6432029) (based on a [2017 publication](https://academic.oup.com/bioinformatics/article/33/21/3387/3931857)) and a [2022 publicaiton](https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gkac278/6576357) which aim at predicting protein subcellular location.

The possible subcellular localizations (in the splits, assigned to `TARGET`) are `Cytoplasm`, `Nucleus`, `Cell membrane`, `Mitochondrion`, `Endoplasmic reticulum`, `Lysosome/Vacuole`, `Golgi apparatus`, `Peroxisome`, `Extracellular` and `Plastid`.

### Full dataset

The `deeploc_our_train_set.fasta`, `deeploc_our_val_set.fasta`, `deeploc_our_test_set.fasta` and `setHARD.fast` are the source dataset from [the Light Attention (LA) 2021 publication](https://academic.oup.com/bioinformaticsadvances/article/1/1/vbab035/6432029) used to create the DeepLoc 1.0 splits. The `human_sequences.tsv` is used to recover the human proteins from the DeepLoc 1.0 set.

The `Swissprot_Train_Validation_dataset.csv` and the `hpa_testset.csv` are the training/validation and test set from the [DeepLoc v2 2022 publicaiton](https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gkac278/6576357) used to create the DeepLoc 2.0 split.

Due to the size of these files, they can be found at http://data.bioembeddings.com/public/FLIP/scl/.

### Splits

All splits are classification splits as explained prior. Train/Test splits are done as follows.

Splits ([semaphore legend](../../README.md#split-semaphore)):
- From DeepLoc 1.0:
    - 🟢 `mixed_soft`: DeepLoc 1.0 Train + DeepLoc 1.0 Validation + DeepLoc 1.0 Test
    - 🟢 `mixed_hard`: DeepLoc 1.0 Train + DeepLoc 1.0 Validation + testHARD Test
    - 🟢 `human_soft`: DeepLoc 1.0 train + DeepLoc 1.0 validation + DeepLoc 1.0 test (only human proteins)
    - 🟢 `human_hard`: DeepLoc 1.0 train + DeepLoc 1.0 validation + DeepLoc 1.0 test (only human proteins)
    - 🟢 `balanced`: Same as `mixed_hard` but with the test set classes balanced with respect to the training set
- From DeepLoc 2.0:
    - 🟢 `mixed_vs_human_2`: DeepLoc 2.0 Train (mixed, multi-label) + DeepLoc 2.0 Validation (mixed, multi-label) + DeepLoc 2.0 test (human, multi-label)

All splits are contained in the `splits.zip` file. These are CSV with colums:

- `sequence`: the AA sequence. May contain special characters!
- `target`: the prediction target, which is string class vlaue. For `mixed_vs_human_2`, the different classes are separated with semicolon.
- `set`: either `train` or `test`, if the sequence should be used for training or testing your model!
- `validation`: When True, these are sequences for train that may be used for validation (e.g. early stopping).


### Cite
From the publishers as Bibtex:
```
@article{10.1093/bioadv/vbab035,
author = {Stärk, Hannes and Dallago, Christian and Heinzinger, Michael and Rost, Burkhard},
title = "{Light attention predicts protein location from the language of life}",
journal = {Bioinformatics Advances},
volume = {1},
number = {1},
year = {2021},
month = {11},
abstract = "{Although knowing where a protein functions in a cell is important to characterize biological processes, this information remains unavailable for most known proteins. Machine learning narrows the gap through predictions from expert-designed input features leveraging information from multiple sequence alignments (MSAs) that is resource expensive to generate. Here, we showcased using embeddings from protein language models for competitive localization prediction without MSAs. Our lightweight deep neural network architecture used a softmax weighted aggregation mechanism with linear complexity in sequence length referred to as light attention. The method significantly outperformed the state-of-the-art (SOTA) for 10 localization classes by about 8 percentage points (Q10). So far, this might be the highest improvement of just embeddings over MSAs. Our new test set highlighted the limits of standard static datasets: while inviting new models, they might not suffice to claim improvements over the SOTA.The novel models are available as a web-service at http://embed.protein.properties. Code needed to reproduce results is provided at https://github.com/HannesStark/protein-localization. Predictions for the human proteome are available at https://zenodo.org/record/5047020.Supplementary data are available at Bioinformatics Advances online.}",
issn = {2635-0041},
doi = {10.1093/bioadv/vbab035},
url = {https://doi.org/10.1093/bioadv/vbab035},
note = {vbab035},
eprint = {https://academic.oup.com/bioinformaticsadvances/article-pdf/1/1/vbab035/41640353/vbab035.pdf},
}
```

```
@article{10.1093/bioinformatics/btx431,
author = {Almagro Armenteros, José Juan and Sønderby, Casper Kaae and Sønderby, Søren Kaae and Nielsen, Henrik and Winther, Ole},
title = "{DeepLoc: prediction of protein subcellular localization using deep learning}",
journal = {Bioinformatics},
volume = {33},
number = {21},
pages = {3387-3395},
year = {2017},
month = {07},
abstract = "{The prediction of eukaryotic protein subcellular localization is a well-studied topic in bioinformatics due to its relevance in proteomics research. Many machine learning methods have been successfully applied in this task, but in most of them, predictions rely on annotation of homologues from knowledge databases. For novel proteins where no annotated homologues exist, and for predicting the effects of sequence variants, it is desirable to have methods for predicting protein properties from sequence information only.Here, we present a prediction algorithm using deep neural networks to predict protein subcellular localization relying only on sequence information. At its core, the prediction model uses a recurrent neural network that processes the entire protein sequence and an attention mechanism identifying protein regions important for the subcellular localization. The model was trained and tested on a protein dataset extracted from one of the latest UniProt releases, in which experimentally annotated proteins follow more stringent criteria than previously. We demonstrate that our model achieves a good accuracy (78\\% for 10 categories; 92\\% for membrane-bound or soluble), outperforming current state-of-the-art algorithms, including those relying on homology information.The method is available as a web server at http://www.cbs.dtu.dk/services/DeepLoc. Example code is available at https://github.com/JJAlmagro/subcellular\_localization. The dataset is available at http://www.cbs.dtu.dk/services/DeepLoc/data.php.}",
issn = {1367-4803},
doi = {10.1093/bioinformatics/btx431},
url = {https://doi.org/10.1093/bioinformatics/btx431},
eprint = {https://academic.oup.com/bioinformatics/article-pdf/33/21/3387/25166063/btx431.pdf},
}
```

```
@article{10.1093/nar/gkac278,
author = {Thumuluri, Vineet and Almagro Armenteros, José Juan and Johansen, Alexander Rosenberg and Nielsen, Henrik and Winther, Ole},
title = "{DeepLoc 2.0: multi-label subcellular localization prediction using protein language models}",
journal = {Nucleic Acids Research},
year = {2022},
month = {04},
abstract = "{The prediction of protein subcellular localization is of great relevance for proteomics research. Here, we propose an update to the popular tool DeepLoc with multi-localization prediction and improvements in both performance and interpretability. For training and validation, we curate eukaryotic and human multi-location protein datasets with stringent homology partitioning and enriched with sorting signal information compiled from the literature. We achieve state-of-the-art performance in DeepLoc 2.0 by using a pre-trained protein language model. It has the further advantage that it uses sequence input rather than relying on slower protein profiles. We provide two means of better interpretability: an attention output along the sequence and highly accurate prediction of nine different types of protein sorting signals. We find that the attention output correlates well with the position of sorting signals. The webserver is available at services.healthtech.dtu.dk/service.php?DeepLoc-2.0.}",
issn = {0305-1048},
doi = {10.1093/nar/gkac278},
url = {https://doi.org/10.1093/nar/gkac278},
note = {gkac278},
eprint = {https://academic.oup.com/nar/advance-article-pdf/doi/10.1093/nar/gkac278/43515314/gkac278.pdf},
}
```

### Data licensing

The RAW data downloaded from both aforementioned publications is subject to [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Modified data available in this repository and in the `splits` falls under [AFL-3](https://opensource.org/licenses/AFL-3.0).


This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/), which permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.