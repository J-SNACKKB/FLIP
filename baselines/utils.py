
import numpy as np
import pandas as pd 
import sys 
import re
from pathlib import Path
from .filepaths import * 
from typing import List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset

vocab = 'ARNDCQEGHILKMFPSTWYVXU'
pad_index = len(vocab)

class Tokenizer(object):
    """convert between strings and their one-hot representations"""
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


class ASCollater(object):
    def __init__(self, alphabet: str, tokenizer: object, pad=False, pad_tok=0., backwards=False):
        self.pad = pad
        self.pad_tok = pad_tok
        self.tokenizer = tokenizer
        self.backwards = backwards
        self.alphabet = alphabet

    def __call__(self, batch: List[Any], ) -> List[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        sequences = [torch.tensor(self.tokenizer.tokenize(s)) for s in sequences]
        sequences = [i.view(-1,1) for i in sequences]
        maxlen = max([i.shape[0] for i in sequences])
        padded = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]),"constant", self.pad_tok) for i in sequences]
        padded = torch.stack(padded)
        mask = [torch.ones(i.shape[0]) for i in sequences]
        mask = [F.pad(i, (0, maxlen - i.shape[0])) for i in mask]
        mask = torch.stack(mask)
        y = data[1]
        y = torch.tensor(y).unsqueeze(-1)
        ohe = []
        for i in padded:
            i_onehot = torch.FloatTensor(maxlen, len(self.alphabet))
            i_onehot.zero_()
            i_onehot.scatter_(1, i, 1)
            ohe.append(i_onehot)
        padded = torch.stack(ohe)
            
        return padded, y, mask

def ridge_one_hot_encoding(s, length, vocab=vocab):
    """one hot encodes seqs for ridge regression"""
    #ll_train = list(ds)
    #seq = [i[0] for i in all_train]
    #target = [i[1] for i in all_train] 
    #tokenizer = Tokenizer(vocab) # tokenize
    #s = [torch.tensor(tokenizer.tokenize(i)).view(-1, 1) for i in s]
    #s = [F.pad(i, (0, 0, 0, length - i.shape[0]), "constant", 0.) for i in s]
    s_enc = [] # one 
    for i in s:
        i_onehot = torch.FloatTensor(length, len(vocab))
        i_onehot.zero_()
        i_onehot.scatter_(1, i, 1)
        s_enc.append(i_onehot)
    s_enc = np.array([np.array(i.view(-1)) for i in s_enc]) # flatten
    return s_enc 


def encode_pad_seqs(s, length, vocab=vocab):
    """pads all sequences, converts AA string to np.array of indices"""
    aa_dict = {k: v for v, k in enumerate(vocab)}
    result = np.full((length), pad_index) 
    for i,l in enumerate(s):
        index = aa_dict[l]
        result[i] = index
    return result


def one_hot_pad_seqs(s, length, vocab=vocab):
    aa_dict = {k: v for v, k in enumerate(vocab)}
    embedded = np.zeros([length, len(vocab)])
    for i, l in enumerate(s):
        idx = aa_dict[l] 
        embedded[i, idx] = 1 
    embedded = embedded.flatten()
    return embedded


def get_data(df, max_length, encode_pad=True, zip_dataset=True, reverse_seq_target=False, one_hots=False): 
    """returns encoded and padded sequences with targets"""
    target = df.target.values.tolist()
    seq = df.sequence.values.tolist() 
    if encode_pad: 
        seq = [encode_pad_seqs(s, max_length) for s in seq]
        print('encoded and padded all sequences to length', max_length)

    if one_hots:
        seq = [one_hot_pad_seqs(s, max_length) for s in seq]
        print('one-hot encoded and padded all sequences to length', max_length)
        print('flattened one-hot sequences')
        return np.array(seq), np.array(target)

    if zip_dataset:
        if reverse_seq_target:
            return list(zip(target, seq))
        else:
            return list(zip(seq, target))
    else: 
        return torch.FloatTensor(seq), torch.FloatTensor(target)

def load_dataset(dataset, split, val_split = True, gb1_shorten=False):
    """returns dataframe of train, (val), test sets, with max_length param"""
    
    datadir = Path(DATA_DIR)

    PATH = datadir / dataset / 'splits' / split 
    print('reading dataset:', split)
        
    df = pd.read_csv(PATH)

    if dataset == 'gb1' and gb1_shorten == True:
        print('shortening gb1 to first 56 AAs')
        df.sequence = df.sequence.apply(lambda s: s[:56])
    
    df.sequence = df.sequence.apply(lambda s: re.sub(r'[^A-Z]', '', s.upper())) #remove special characters
    max_length = max(df.sequence.str.len())
    
    if val_split == True:
        test = df[df.set == 'test']
        train = df[(df.set == 'train')&(df.validation.isna())] # change False for meltome 
        val = df[df.validation == True]

        print('loaded train/val/test:', len(train), len(val), len(test))
        return train, val, test, max_length
    else:
        test = df[df.set == 'test']
        train = df[(df.set == 'train')]
        print('loaded train/test:', len(train), len(test))
        return train, test, max_length


def load_esm_dataset(dataset, model, split, mean, mut_mean, flip, gb1_shorten=False):

    embedding_dir = Path(EMBEDDING_DIR)
    PATH = embedding_dir / dataset / model / split
    print('loading ESM embeddings:', split)
    
    if mean:
        train = torch.load(PATH / 'train_mean.pt') #data_len x seq x 1280
        val = torch.load(PATH / 'val_mean.pt')
        test = torch.load(PATH / 'test_mean.pt') #data_len x seq x 1280
    else:
        train = torch.load(PATH / 'train_aa.pt') #data_len x seq x 1280
        val = torch.load(PATH / 'val_aa.pt')
        test = torch.load(PATH / 'test_aa.pt') #data_len x seq x 1280

        if dataset == 'gb1' and gb1_shorten == True: #fix the sequence to be shorter
            print('shortening gb1 to first 56 AAs')
            train = train[:, :56, :]
            val = val[:, :56, :]
            test = test[:, :56, :]
    
    if dataset == 'aav' and mut_mean == True:
        train = torch.mean(train[:, 560:590, :], 1)
        val = torch.mean(val[:, 560:590, :], 1)
        test = torch.mean(test[:, 560:590, :], 1)

    if dataset == 'gb1' and mut_mean == True: #positions 39, 40, 41, 54 in sequence
        train = torch.mean(train[:, [38, 39, 40, 53], :], 1)
        val = torch.mean(val[:, [38, 39, 40, 53], :], 1)
        test = torch.mean(test[:, [38, 39, 40, 53], :], 1)
    

    train_l = torch.load(PATH / 'train_labels.pt')
    val_l = torch.load(PATH / 'val_labels.pt')
    test_l = torch.load(PATH / 'test_labels.pt')

    if flip:
        train_l, test_l = test_l, train_l 
        train, test = test, train
   
    train_esm_data = TensorDataset(train, train_l)
    val_esm_data = TensorDataset(val, val_l)
    test_esm_data = TensorDataset(test, test_l)

    max_length = test.shape[1]

    print('loaded train/val/test:', len(train_esm_data), len(val_esm_data), len(test_esm_data), file = sys.stderr) 
    
    return train_esm_data, val_esm_data, test_esm_data, max_length


class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        row = self.data.iloc[index]
        return row['sequence'], row['target']

class ESMSequenceDataset(Dataset):
    "special dataset class just to deal with ESM tensors"
    def __init__(self, emb, mask, labels):
        self.emb = emb
        self.mask = mask
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.emb[index], self.mask[index], self.labels[index]

class HugeDataset(Dataset):
    "load in the data directly from saved .pt files output from batch ESM. Include test/train in path"
    def __init__(self, embeddings_path, label_path, mean=False):
        self.path = embeddings_path
        self.label = torch.load(label_path)
        self.mean = mean
    
    def __len__(self):
        return self.label.shape[0]
    
    def __getitem__(self, index):
        if self.mean:
            e = torch.load(self.path + str(index) + '.pt')['mean_representations'][33]
        else:   
            e = torch.load(self.path + str(index) + '.pt')['representations'][33]
        
        return e, self.label[index]

