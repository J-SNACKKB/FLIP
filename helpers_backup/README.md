# Helpers

This folder contains helper scripts and modules for FLIP. Each script and module has a specific purpose, from handling data loading and preprocessing to defining the neural network model.

## Contents

- `dataloader.py`: Defines a custom Dataset class and functions to create a DataLoader.
- `models_flip.py`: Defines the Convolutional Neural Network (CNN) used for the baseline.
- `sequence_database.py`: Defines a SequenceDatabase class to store and manage protein sequences, their targets, and annotations.
- `sequence_encoding.py`: Contains functions to encode protein sequences using different methods.
- `extract_small_csv_difflength.py`: A script to create a smaller CSV file with sequences of different lengths for testing purposes.
- `four_mutations_small.csv`: A sample data file.
- `four_mutations_random_lengths.csv`: A sample data file with sequences of varying lengths for testing purposes.

## Usage

### Environment Setup - micromamba 

Ensure you have the necessary dependencies installed (listed in the environment.yml file) by following the instructions to prepare your environment:

```bash
micromamba create -f environment_helpers.yml
micromamba activate flip
