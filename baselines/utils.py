import pandas as pd
import numpy as np
import random
import re
from pathlib import Path
import sys

import torch
from torch.utils.data import Dataset, TensorDataset

vocab = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
pad_index = len(vocab) # pad index is 20


def encode_pad_seqs(s, length, vocab = vocab):
    """pads all sequences, converts AA string to np.array of indices"""
    aa_dict = {k: v for v, k in enumerate(vocab)}
    result = np.full((length), pad_index) 
    for i,l in enumerate(s):
        index = aa_dict[l]
        result[i] = index
    return result

def get_data(df, max_length, encode_pad=True, zip_dataset=True, reverse_seq_target=False): 
    """returns encoded and padded sequences with targets"""
    target = df.target.values.tolist()
    seq = df.sequence.values.tolist() 
    if encode_pad: 
        seq = [encode_pad_seqs(s, max_length) for s in seq]
        print('encoding and padding all sequences to length', max_length)
    if zip_dataset:
        if reverse_seq_target:
            return list(zip(target, seq))
        else:
            return list(zip(seq, target))
    else: 
        return torch.FloatTensor(seq), torch.FloatTensor(target)

def load_dataset(dataset, split, val_split = True):
    """returns dataframe of train, (val), test sets, with max_length param"""
    
    datadir = '/../../FLIP/splits/'+dataset+'/splits/'

    path = datadir+split
    print('reading dataset:', split)
        
    df = pd.read_csv(path)
    
    df.sequence.apply(lambda s: re.sub(r'[^A-Z]', '', s.upper())) #remove special characters
    max_length = max(df.sequence.str.len())
    
    if val_split == True:
        test = df[df.set == 'test']
        train = df[(df.set == 'train')&(df.validation.isnull())]
        val = df[df.validation == True]

        print('loaded train/val/test:', len(train), len(val), len(test))
        return train, val, test, max_length
    else:
        test = df[df.set == 'test']
        train = df[(df.set == 'train')]
        print('loaded train/test:', len(train), len(test))
        return train, test, max_length


def load_esm_dataset(dataset, model, split, mean, mut_mean, samples, index, flip):

    embedding_dir = Path('/../../embeddings/')
    PATH = embedding_dir / dataset / model / split
    
    if mean:
        train = torch.load(PATH / 'train_mean.pt') #data_len x seq x 1280
        val = torch.load(PATH / 'val_mean.pt')
        test = torch.load(PATH / 'test_mean.pt') #data_len x seq x 1280
    else:
        train = torch.load(PATH / 'train_aa.pt') #data_len x seq x 1280
        val = torch.load(PATH / 'val_aa.pt')
        test = torch.load(PATH / 'test_aa.pt') #data_len x seq x 1280
    
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

    # TEMPORARY FIX TODO: resave without the zeros!!!
    #test = test[:test_l.shape[0]]

    if flip:
        train_l, test_l = test_l, train_l 
        train, test = test, train

    if index is not None:

        train = train[samples[index]]
        train_l = train_l[samples[index]]
   
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

