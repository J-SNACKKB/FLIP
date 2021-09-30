# Baseline Description

We benchmark our datasets and splits on 3 types of baselines:
- Pretrained: ESM-1b, ESM-1v
- Supervised: Ridge regression, 
- Parameter-free: Levenshtein distance, BLOSUM62

Further details about our baselines are below: 

| Model                    | Description                                              | Encoding                               | Sequence Dimension Pooling   | Output Layer   | File                                   |
|--------------------------|----------------------------------------------------------|----------------------------------------|------------------------------|----------------|----------------------------------------|
| Levenshtein              | Levenshtein & Levenshtein distance to wild-type.         | -                                      | -                            | -              | `levenshtein_blosum62_baselines.ipynb` |
| BLOSUM62                 | BLOSUM62-score relative to wild-type.                    | -                                      | -                            | -              | `levenshtein_blosum62_baselines.ipynb` |
| Ridge regression         | Ridge regression model                                   | One-hot (20 x seq)                     | -                            | 1 Dense Layer  | `linear_models.py`                     |
| CNN                      | Simple convolutional network                             | One-hot (20 x seq)                     | 1D Convolution               | 1 Dense Layer  | `cnn.py`                               |
| ESM-1b                   | 750M param transformer pretrained on Uniref 50           | per-AA ESM embeddings (1280 x seq)     | 1D Attention                 | 2 Dense Layers | `train_all.py`                         |
| ESM-1b (mean)            | 750M param transformer pretrained on Uniref 50           | per-sequence ESM embeddings (1280 x 1) | Mean across entire sequence  | 2 Dense Layers | `train_all.py`                         |
| ESM-1b (mut mean)        | 750M param transformer pretrained on Uniref 50           | per-AA ESM embeddings (1280 x seq)     | Mean across mutated residues | 2 Dense Layers | `train_all.py`                         |
| ESM-1v                   | 750M param transformer pretrained on Uniref 90           | per-AA ESM embeddings (1280 x seq)     | 1D Attention                 | 2 Dense Layers | `train_all.py`                         |
| ESM-1v (mean)            | 750M param transformer pretrained on Uniref 90           | per-sequence ESM embeddings (1280 x 1) | Mean across entire sequence  | 2 Dense Layers | `train_all.py`                         |
| ESM-1v (mut mean)        | 750M param transformer pretrained on Uniref 90           | per-AA ESM embeddings (1280 x seq)     | Mean across mutated residues | 2 Dense Layers | `train_all.py`                         |
| ESM-untrained            | 750M param transformer with randomly initialized weights | per-AA ESM embeddings (1280 x seq)     | 1D Attention                 | 2 Dense Layers | `train_all.py`                         |
| ESM-untrained (mean)     | 750M param transformer with randomly initialized weights | per-sequence ESM embeddings (1280 x 1) | Mean across entire sequence  | 2 Dense Layers | `train_all.py`                         |
| ESM-untrained (mut mean) | 750M param transformer with randomly initialized weights | per-AA ESM embeddings (1280 x seq)     | Mean across mutated residues | 2 Dense Layers | `train_all.py`                         |

## ESM

All datasets need to first be embedded using the appropriate ESM model (ESM-1b, ESM-1v, ESM-untrained). We provide a script, `embed.py`, that 1) performs bulk ESM embeddings using pretrained models and 2) saves concatenated PyTorch tensors for train, test, and validation splits. More information on ESM embeddings can be found at the original [ESM repo](https://github.com/facebookresearch/esm).

Once embedded and saved in `.pt` format, ESM models can be run using the `train_all.py` script. For example, the following command trains the des-mut AAV split on ESM-1b using GPU 3:
 ```$ python train_all.py aav_1 esm1b 3```

The following shorthands for splits are  used in running the scripts:
`aav_1`: `des_mut`
`aav_1`: `des_mut` ,
`aav_2`: `mut_des`,
`aav_3`: `one_vs_many`,
`aav_4`: `two_vs_many`,
`aav_5`: `seven_vs_many`,
`aav_6`: `low_vs_high`,
`aav_7`: `sampled`,
`meltome_1` : `mixed_split`,
`meltome_2`: `human`,
`meltome_3` : `human_cell`,
`gb1_1`: `one_vs_rest`,
`gb1_2`: `two_vs_rest`,
`gb1_3`: `three_vs_rest`,
`gb1_4`: `sampled`,
`gb1_5`: `low_vs_high`

### Arguments for `train_all.py`:
- `split` choose from shorthand above
- `model` choose from `esm1b`, `esm1v`, `esm_rand` (ESM-untrained)

Optional:
- `--mean` take the mean across all sequences 
- `--mut_mean` take the mean in the mutated regions only (only applicable for GB1 and AAV)
- `--lr 0.001` set the learning rate - default 0.001
- `--ensemble` run the model 10 times with different seeds


## CNN 

The CNN model does not require any embedding. To run the des-mut AAV split on GPU 3:
```$ python cnn.py aav aav_5 /path/to/model/save/ --gpu 3```

### Arguments for `cnn.py`:
- `dataset` choose from aav, gb1, meltome
- `split` choose from shorthand above
- `/path/to/model/save/` where to save trained model; used later in testing
- `--gpu #` GPU to use

Optional:
- `--kernel_size` kernel size of the 1D CNN, default 5
- `--input_size` input channels of the CNN; outputs are 2xinput, default 1024
- `--dropout` amount of dropout, default 0.0
- `--ensemble` run the model 10 times with different seeds

## Ridge

To run ridge regression on des-mut AAV split:
- ```$ python linear_models.py aav aav_1```

### Arguments for `linear_models.py`:
- `dataset` choose from aav, gb1, meltome
- `split` choose from shorthand above

Optional:
- `--alpha` regularization strength 

## Levenshtein and BLOSUM62

Both of these can be run in the `levenshtein_blosum62_baselines.ipynb` notebook using the Biopython package.

## Results

Results from all baselines will be saved to a results file, e.g. `aav_results.csv` with the following headers: [dataset, model, split, index, train rho, train MSE, test rho, test MSE, epochs trained, hyperparameters used]
