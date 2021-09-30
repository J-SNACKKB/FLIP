import argparse
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

from train_all import split_dict 
from utils import SequenceDataset, load_dataset
from csv import writer
from pathlib import Path

class CSVDataset(Dataset):

    def __init__(self, fpath=None, df=None, split=None, outputs=[]):
        if df is None:
            self.data = pd.read_csv(fpath)
        else:
            self.data = df
        if split is not None:
            self.data = self.data[self.data['split'] == split]
        self.outputs = outputs
        self.data = self.data[['sequence'] + self.outputs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return [row['sequence'], *row[self.outputs]]


class Tokenizer(object):
    """Convert between strings and their one-hot representations."""
    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.a_to_t = {a: i for i, a in enumerate(self.alphabet)}
        self.t_to_a = {i: a for i, a in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)

    def tokenize(self, seq: str) -> np.ndarray:
        return np.array([self.a_to_t[a] for a in seq])

    def untokenize(self, x) -> str:
        return ''.join([self.t_to_a[t] for t in x])

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='file path to data directory')
parser.add_argument('task', type=str)
parser.add_argument('--scale', type=bool, default=False)
parser.add_argument('--solver', type=str, default='lsqr')
parser.add_argument('--max_iter', type=float, default=1e6)
parser.add_argument('--tol', type=float, default=1e-4)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--ensemble', action='store_true')

args = parser.parse_args()

AAINDEX_ALPHABET = 'ARNDCQEGHILKMFPSTWYVXU'
# grab data
split = split_dict[args.task]
train, test, _ = load_dataset(args.dataset, split+'.csv', val_split=False)

ds_train = SequenceDataset(train)
ds_test = SequenceDataset(test)


# tokenize train data
all_train = list(ds_train)
X_train = [i[0] for i in all_train]
y_train = [i[1] for i in all_train]
if args.dataset == 'meltome':
    AAINDEX_ALPHABET += 'XU'
tokenizer = Tokenizer(AAINDEX_ALPHABET) # tokenize
X_train = [torch.tensor(tokenizer.tokenize(i)).view(-1, 1) for i in X_train]


# tokenize test data
all_test = list(ds_test)
X_test = [i[0] for i in all_test]
y_test = [i[1] for i in all_test]
tokenizer = Tokenizer(AAINDEX_ALPHABET) # tokenize
X_test = [torch.tensor(tokenizer.tokenize(i)).view(-1,1) for i in X_test]


# padding
maxlen_train = max([len(i) for i in X_train])
maxlen_test = max([len(i) for i in X_test])
maxlen = max([maxlen_train, maxlen_test])
# pad_tok = alphabet.index(PAD)

X_train = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]), "constant", 0.) for i in X_train]
X_train_enc = [] # ohe
for i in X_train:
    i_onehot = torch.FloatTensor(maxlen, len(AAINDEX_ALPHABET))
    i_onehot.zero_()
    i_onehot.scatter_(1, i, 1)
    X_train_enc.append(i_onehot)
X_train_enc = np.array([np.array(i.view(-1)) for i in X_train_enc]) # flatten

X_test = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]),"constant", 0.) for i in X_test]
X_test_enc = [] # ohe
for i in X_test:
    i_onehot = torch.FloatTensor(maxlen, len(AAINDEX_ALPHABET))
    i_onehot.zero_()
    i_onehot.scatter_(1, i, 1)
    X_test_enc.append(i_onehot)
X_test_enc = np.array([np.array(i.view(-1)) for i in X_test_enc]) # flatten


# scale X
if args.scale:
    scaler = StandardScaler()
    X_train_enc = scaler.fit_transform(X_train_enc)
    X_test_enc = scaler.transform(X_test_enc)

def main(args, X_train_enc, y_train, y_test):


    print('Parameters...')
    print('Solver: %s, MaxIter: %s, Tol: %s' % (args.solver, args.max_iter, args.tol))
    print('Training...')
    lr = Ridge(solver=args.solver, tol=args.tol, max_iter=args.max_iter,alpha=args.alpha)
    # lr = LinearRegression()
    lr.fit(X_train_enc, y_train)
    preds = lr.predict(X_test_enc)
    mse = mean_squared_error(y_test, preds)
    rho = spearmanr(y_test, preds).correlation
    print('TEST MSE: ', mse)
    print('TEST RHO: ', rho)

    with open(Path.cwd() / 'evals_new'/ (args.dataset+'_results.csv'), 'a', newline='') as f:
        writer(f).writerow([args.dataset, 'linear', split, '', '', '', round(rho,2), round(mse,2), '', args.alpha])


if args.ensemble:
    for i in range(10):
        np.random.seed(i)
        torch.manual_seed(i)
        main(args, X_train_enc, y_train, y_test)
else:
    np.random.seed(0)
    torch.manual_seed(1)
    main(args, X_train_enc, y_train, y_test)