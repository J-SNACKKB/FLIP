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
│   └───advenced_environment.yml <-- File with the required environment for the execution OR some description about poetry/docker etc.
│   │
│   └───train_and_eval.py <--------- Here goes the script for the training and evaluation of the splits
│   │
│   └───utils.py <------------------- Here go auxiliary functions (if required)
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
- `split`: name of the split to be evaluated from a list of available splits. Used to obtain tmp files required following the requirements of biotrainer (to be decided/standarized). E.g. could be `sequence_file.fasta` and `labels_file.fasta`. Also to select the database of origin used to select the corresponding config file.
- `embeddings`: name of the embeddings to be used from a list of available embeddings.
- `embeddings_file_path`: path to the embeddings file.
- The same as in biotrainer / offered in the configuration files for possible modifications of the dataset associated config.

### Schema of the script

1. Get the arguments  
  1.1. If `--config` is not provided, use the default config file in `configsbank/`. If provided, use the provided one.
2. [FOR FUTURE VERSIONS] Look for possible incompatibiliest between the configuration file (if it is not the default) and the selected split
3. Prepare the data of the split as expected by biotrainer (`sequence_file` and `labels_file`)
4. Create (temporal) copy of the configuration file with the path to the data of the selected split and the selected embedding
5. Create folder to allocate the results
6. Pass the control to biotrainer with a call using poetry/docker

## About `utils.py` file:

TO DO


## About `configsbank` folder:
- The `configsbank` folder contains the expected and recomended configuration files for the different datasets.
- The configuration files are named after the dataset: `config_<dataset_name>.txt`.
- The configuration files follow the format expected by biotrainer, e.g.:
```yaml
sequence_file: sequences.fasta # Specify your sequence file
labels_file: labels.fasta # Specify your label file
protocol: residue_to_class # residue_to_class | sequence_to_class : Prediction method
model_choice: CNN # CNN | ConvNeXt | FNN | LogReg : Prediction model 
optimizer_choice: adam # adam : Model optimizer
loss_choice: cross_entropy_loss # cross_entropy_loss : Model loss 
num_epochs: 200 # 1-n : Number of maximum epochs
use_class_weights: False # Balance class weights by using class sample size in the given dataset
learning_rate: 1e-3 # 0-n : Model learning rate
batch_size: 128 # 1-n : Batch size
embedder_name: prottrans_t5_xl_u50 # one_hot_encoding | word2vec | prottrans_t5_xl_u50 | ... : Sequence embedding method (see below)
embeddings_file: /path/to/embeddings.h5 # optional, if defined will use 'embedder_name' to name experiment
```
  - `sequence_file` and `labels_file` should be empty and completed once the user specify the dataset/split to be tested.
  - The rest of the parameters must include a basic/recomended preset value.



