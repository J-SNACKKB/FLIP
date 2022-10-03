# Bio-Benchmarks for Protein Engineering

This repository is for the paper submitted to the 2021 NeurIPS Benchmark track.


## Folder breakup

1. `collect_splits` contains notebooks to process RAW datasets collected from various sources.
1. `splits` contains all splits, a brief description of their processing and the logic behind train/test splits
1. `baselines` contains code used to compute baselines

A `.gitignore`d folder called `data` contains RAW data used to produce all splits. As the folder size is substantial, it could not be shipped with GitHub. However, it can be accessed here: http://data.bioembeddings.com/public/FLIP

[Here](http://data.bioembeddings.com/public/FLIP/fasta/) are available all the FLIP datasets in FASTA format (following the standardization proposed in [biotrainer](https://github.com/sacdallago/biotrainer)).

## Find out more about the splits

The goal of the splits in this repository is to assess how well machine learning devices using protein sequence inputs can represent different dimensions relevant for protein design.
The main place to find out about the splits is the `splits` folder. Each set contains a zip file with one or more "splits", where different splits may be different train/test splits based on biological or statistical intuition.

### Split semaphore
Splits are associated with a semaphore which indicates for what they may be used:

- 🟢: _active_ splits can be used to evaluate accuracy of your machine learning models
- 🟠: splits that should not be used to make performance comparisons, as may give overestimations, or because other active splits have similar discriminative ability
- 🔴: splits that should not be used / considered obsolete. Please do not use these to report performance.
