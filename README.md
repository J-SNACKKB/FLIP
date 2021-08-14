# Bio-Benchmarks for Protein Engineering

This repository is for the paper submitted to the 2021 NeurIPS Benchmark track.


## Folder breakup

1. `collect_tasks` contains notebooks to process RAW datasets collected from various sources.
1. `tasks` contains all tasks, a brief description of their processing and the logic behind train/test splits
1. `baselines` contains code used to compute baselines

A `.gitignore`d folder called `data` contains RAW data used to produce all tasks.
As the folder size is substantial, and additional data to the RAW sets was used in some instances, this folder is not shipped with the repository but available on-demand.

## Find out more about the tasks

The goal of the tasks in this repository is to assess how well machine learning devices using protein sequence inputs can represent different dimensions relevant for protein design.
The main place to find out about the tasks is the `tasks` folder. Each set contains a zip file with one or more "tasks", where different tasks may be different train/test splits based on biological or statistical intuition.

Note that we include both 🟢 ACTIVE and 🔴 DEPRECATED sets. Evaluations should be performed only on the 🟢 ACTIVE sets.