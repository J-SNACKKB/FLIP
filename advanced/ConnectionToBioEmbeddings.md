# Schema for the connection of FLIP with bio-embeddings (via biotrainer)

Proposed tree of folders:

```
FLIP
│   ...    
│
└───baselines/
│
└───collect_splots/
│
└───data/ (not in GitHub)
│
└───helpers/
│
└───splits/
│
└───advanced/ <---------------------- Here goes the connection to bio-embeddings (using biotrainer) (Name for the folder?)
│   │
│   └───train_and_eval.py <--------- Here goes the script for the training and evaluation of the splits
│   │
│   └───utils.py <------------------- Here go auxiliary functions
│   │   
│   └───configsbank/ <-------------- Here go the configuration files (recommended, basic and default config) for the different splits
│   │   
│   │   config_aav.txt
│   │   config_gb1.txt
│   │   config_meltome.txt
│   │   ...
│   │
```

## About `train_and_eval.py` file:

### Arguments
- `split`: name of the split to be evaluated from a list of available splits. Used to obtain required tmp files following the requirements of biotrainer. E.g. could be `sequence_file.fasta` and/or `labels_file.fasta`. Also used to select the correct config file from configsbank.
- `protocol`: protocol to be used for the training from a list of available protocols. Used to know how to manage the data transformation form CSV to the required by biotrainer FASTA files.
- `--embedder`: name of the embedder to be used from a list of available embedders.
- `--config`: dir of a config file different from the provided (template) one in configsbank.

Havin just one template config file, and with `--embedder` and `--model` arguments, we can modify the same config template for different executions, remaining other biotrainer arguments unchanged.

### Schema of the script

1. Get the arguments
  1.1. If `--config` is not provided, use the default config file in `configsbank/`. If provided, use the provided one.
2. Prepare the output folder and data.
  2.1. Create the output folder.
  2.2. Create copy of the config template file in the output folder.
  2.3. If available in FASTA format, get FASTA files of the selected split (in biotrainer standardization). Otherwise, convert the FLIP CSV files for the given split to required by biotrainer FASTA files. Save the FASTA files in the output folder.
  2.4 Modify the config file in the output folder with the provided arguments and paths to the FASTA files.
3. Pass the control to biotrainer with a call.

## About `utils.py` file:

It contains some auxiliary functions to be used in the `train_and_eval.py` file:
- `residue_to_class_fasta`, `residue_to_value_fasta`, `protein_to_class_fasta` and `protein_to_value_fasta`: Functions to convert FLIP CSV files into FASTA files, as required by biotrainer.

## About `configsbank` folder:

This folder contains the template/expected/recomended configuration files for the different datasets. The configuration files are named like the datasets, e.g. `config_aav.yml` for the `AAV` splits. These files follow the format expected by biotrainer.