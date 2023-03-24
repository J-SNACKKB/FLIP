### Dataset description

The Conservation split stem from a [2021 publication](https://link.springer.com/article/10.1007/s00439-021-02411-y) which aims at predicting the conservation score of the residues of a protein sequence.

This is a well-known dataset and it is used to validate the behavior of code and models. Only provided a `sampled` split for this purpose.

### Full dataset

The full dataset is divided in `seq_and_conservation.txt`, a FASTA file with 3 lines per entry: amino acid sequence, cotinuous conservation scores (ignored), conservation classes between 1 and 9 (1 = very variable, 9 = very conserved); `train_ids.txt`, IDs of the proteins for training (9392 proteins); `val_ids.txt`, IDs of the proteins for validation (555 proteins); and `test_ids.txt`, IDs of the proteins for testing (519 proteins).

Due to the size of these files, they can be found at http://data.bioembeddings.com/public/FLIP/fasta/conservation/.

### Splits

All splits are classification splits as explained prior. Train/Test splits are done as follows.

Splits ([semaphore legend](../../README.md#split-semaphore)):
- 🟢 `sampled`: Randomly split sequences into `train`/`test` with 95/5% probability.

All splits are contained in the `splits.zip` file. There are one `sequences.fasta` file with all the sequences of the splits in FASTA format and one FASTA file with the labels for each split, in this case only one `sampled.fasta`file.

The labels files are organized by sequence ID. Each sequence label has `SET` atribute (either `train` or `test`) and `VALIDATION` attribute (when True, these are sequences for train that may be used for validation (e.g. early stopping)). Example:
```
>Seq1 SET=train VALIDATION=False
97611111
```

The labels are string encodings of sequences.  For each amino acid there is a conservation class between 1 and 9 (1 = very variable, 9 = very conserved).

### Cite
From the publishers as Bibtex:
```bibtex
@article{marquet2021embeddings,
title={Embeddings from protein language models predict conservation and variant effects},
author={Marquet, C{\'e}line and Heinzinger, Michael and Olenyi, Tobias and Dallago, Christian and Erckert, Kyra and Bernhofer, Michael and Nechaev, Dmitrii and Rost, Burkhard},
journal={Human genetics},
pages={1--19},
year={2021},
abstract = "{The emergence of SARS-CoV-2 variants stressed the demand for tools allowing to interpret the effect of single amino acid variants (SAVs) on protein function. While Deep Mutational Scanning (DMS) sets continue to expand our understanding of the mutational landscape of single proteins, the results continue to challenge analyses. Protein Language Models (pLMs) use the latest deep learning (DL) algorithms to leverage growing databases of protein sequences. These methods learn to predict missing or masked amino acids from the context of entire sequence regions. Here, we used pLM representations (embeddings) to predict sequence conservation and SAV effects without multiple sequence alignments (MSAs). Embeddings alone predicted residue conservation almost as accurately from single sequences as ConSeq using MSAs (two-state Matthews Correlation Coefficient—MCC—for ProtT5 embeddings of 0.596 ± 0.006 vs. 0.608 ± 0.006 for ConSeq). Inputting the conservation prediction along with BLOSUM62 substitution scores and pLM mask reconstruction probabilities into a simplistic logistic regression (LR) ensemble for Variant Effect Score Prediction without Alignments (VESPA) predicted SAV effect magnitude without any optimization on DMS data. Comparing predictions for a standard set of 39 DMS experiments to other methods (incl. ESM-1v, DeepSequence, and GEMME) revealed our approach as competitive with the state-of-the-art (SOTA) methods using MSA input. No method outperformed all others, neither consistently nor statistically significantly, independently of the performance measure applied (Spearman and Pearson correlation). Finally, we investigated binary effect predictions on DMS experiments for four human proteins. Overall, embedding-based methods have become competitive with methods relying on MSAs for SAV effect prediction at a fraction of the costs in computing/energy. Our method predicted SAV effects for the entire human proteome (~ 20 k proteins) within 40 min on one Nvidia Quadro RTX 8000. All methods and data sets are freely available for local and online execution through bioembeddings.com, https://github.com/Rostlab/VESPA, and PredictProtein.}"
publisher={Springer}
}
```

### Data licensing

RAW data was provided by the paper authors and is not subject to license.
Modified data available in this repository and in the `splits` falls under [AFL-3](https://opensource.org/licenses/AFL-3.0).

This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/), which permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.