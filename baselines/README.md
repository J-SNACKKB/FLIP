# Baseline Descriptions

We benchmark our datasets and splits on 3 types of baselines:
- Pretrained: ESM-1b, ESM-1v
- Supervised: Ridge regression, CNN, ESM-untrained
- Parameter-free: Levenshtein distance, BLOSUM62

Further details about our baselines are below: 
| Model                    | Description                                              | Encoding                               | Sequence Dimension Pooling   | Output Layer   |
|--------------------------|----------------------------------------------------------|----------------------------------------|------------------------------|----------------|
| Levenshtein              | Levenshtein & Levenshtein distance to wild-type.         | -                                      | -                            | -              |
| BLOSUM62                 | BLOSUM62-score relative to wild-type.                    | -                                      | -                            | -              |
| Ridge regression         | Ridge regression model                                   | One-hot (20 x seq)                     | -                            | -              |
| CNN                      | Simple convolutional network                             | One-hot (20 x seq)                     | 1D Convolution               | 1 Dense Layer  |
| ESM-1b                   | 750M param transformer pretrained on Uniref 50           | per-AA ESM embeddings (1280 x seq)     | 1D Attention                 | 2 Dense Layers |
| ESM-1b (mean)            | 750M param transformer pretrained on Uniref 50           | per-sequence ESM embeddings (1280 x 1) | Mean across entire sequence  | 2 Dense Layers |
| ESM-1b (mut mean)        | 750M param transformer pretrained on Uniref 50           | per-AA ESM embeddings (1280 x seq)     | Mean across mutated residues | 2 Dense Layers |
| ESM-1v                   | 750M param transformer pretrained on Uniref 90           | per-AA ESM embeddings (1280 x seq)     | 1D Attention                 | 2 Dense Layers |
| ESM-1v (mean)            | 750M param transformer pretrained on Uniref 90           | per-sequence ESM embeddings (1280 x 1) | Mean across entire sequence  | 2 Dense Layers |
| ESM-1v (mut mean)        | 750M param transformer pretrained on Uniref 90           | per-AA ESM embeddings (1280 x seq)     | Mean across mutated residues | 2 Dense Layers |
| ESM-untrained            | 750M param transformer with randomly initialized weights | per-AA ESM embeddings (1280 x seq)     | 1D Attention                 | 2 Dense Layers |
| ESM-untrained (mean)     | 750M param transformer with randomly initialized weights | per-sequence ESM embeddings (1280 x 1) | Mean across entire sequence  | 2 Dense Layers |
| ESM-untrained (mut mean) | 750M param transformer with randomly initialized weights | per-AA ESM embeddings (1280 x seq)     | Mean across mutated residues | 2 Dense Layers |

## ESM (Evolutionary Scale Modeling) Embedding

All datasets need to first be embedded using the appropriate ESM model (ESM-1b, ESM-1v, ESM-untrained). We provide a script, `embeddings.py`, that 1) performs bulk ESM embeddings using pretrained models and 2) saves concatenated PyTorch tensors for train, test, and validation splits. More information on ESM embeddings can be found at the original [ESM repo](https://github.com/facebookresearch/esm). The ESM repo is included as a submodule of this repo, so you can use `git submodule update --init` to populate the `esm` submodule (if you didn't use the `--recurse-submodules` argument when cloning this repo).

Once embedded and saved in `.pt` format, ESM models can be run using the `train_all.py` script. For example, the following command trains the des-mut AAV split on ESM-1b using GPU 3:
 ```$ python train_all.py aav_1 esm1b 3```


## `train_all.py`:

The following shorthands for splits are used in running the scripts:
- `aav_1`: `des_mut`
- `aav_1`: `des_mut` 
- `aav_2`: `mut_des`
- `aav_3`: `one_vs_many`
- `aav_4`: `two_vs_many`
- `aav_5`: `seven_vs_many`
- `aav_6`: `low_vs_high`
- `aav_7`: `sampled`
- `meltome_1` : `mixed_split`
- `meltome_2`: `human`
- `meltome_3` : `human_cell`
- `gb1_1`: `one_vs_rest`
- `gb1_2`: `two_vs_rest`
- `gb1_3`: `three_vs_rest`
- `gb1_4`: `sampled`
- `gb1_5`: `low_vs_high`

### Arguments 

- `split` choose from shorthand above
- `model` choose from `ridge`, `cnn`,`esm1b`, `esm1v`, `esm_rand` (esm untrained)
- `gpu` default 0

Optional:
- `--mean` take the mean across all sequences 
- `--mut_mean` take the mean in the mutated regions only (only applicable for GB1 and AAV)
- `--ensemble` run the model 10 times with different seeds
- `--gb1_shorten` truncate gb1 to one domain, as tested in the dataset paper (will be finalized at later date)

Hyperparameters: 
- `--lr 0.001` learning rate for ESM models - default 0.001
- `--kernel_size` kernel size for CNN models - default 5 
- `--input_size` input size for CNN models - default 1024
- `--dropout` dropout for CNN models - default 0.0
- `--alpha` alpha value for ridge regression - default 1.0


## Levenshtein and BLOSUM62

Both of these can be run in the `levenshtein_blosum62_baselines.ipynb` notebook using the [Biopython](https://biopython.org/) package.

## Results

Results from all baselines will be saved to a results file, e.g. `aav_results.csv` with the following headers: [dataset', 'model', 'split', 'train_rho', 'train_mse', 'test_rho', 'test_mse', 'epochs_trained', 'lr', 'kernel_size', 'input_size', 'dropout', 'alpha', 'gb1_shorten']
