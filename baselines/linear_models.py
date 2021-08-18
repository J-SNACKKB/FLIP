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

torch.manual_seed(0)


AAINDEX_ALPHABET = 'ARNDCQEGHILKMFPSTWYV'


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
args = parser.parse_args()


# grab data
data_fpath = 'tasks/%s/tasks/%s.csv' %(args.dataset, args.task)
df = pd.read_csv(data_fpath)
df = df.astype({'target': float})
df = df.rename({'set': 'split'}, axis='columns')
idx = df[df['split'] == 'train'].index
ds_train = CSVDataset(df=df, split='train', outputs=['target'])
ds_test = CSVDataset(df=df, split='test', outputs=['target'])


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

print('Parameters...')
print('Solver: %s, MaxIter: %s, Tol: %s' % (args.solver, args.max_iter, args.tol))
print('Training...')
lr = Ridge(solver=args.solver, tol=args.tol, max_iter=args.max_iter,)
# lr = LinearRegression()
lr.fit(X_train_enc, y_train)
preds = lr.predict(X_test_enc)
mse = mean_squared_error(y_test, preds)
print('TEST MSE: ', mse)
print('TEST RHO: ', spearmanr(y_test, preds).correlation)