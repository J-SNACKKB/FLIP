import numpy as np
import time
import random

import os
import sys
import torch

import fastaparser
import esm
from tqdm import tqdm
import re

import pandas as pd
from pathlib import Path

import argparse 

sys.path.append('../../FLIP/baselines/')
from utils import load_dataset
from train_all import split_dict

esm_dict = {
    'esm1v': 'esm1v_t33_650M_UR90S_1', # use first of 5 models 
    'esm1b': 'esm1b_t33_650M_UR50S',
    'esm_rand': '/../../random_init_esm.pt' # need to randomly initailize an ESM model and save it 
}

def create_parser():
    parser = argparse.ArgumentParser(
        description="create ESM embeddings"
    )

    parser.add_argument(
        "split",
        type=str
    )

    parser.add_argument(
        "esm_version",
        type = str 
    )

    parser.add_argument(
        "gpu",
        type = str
    )

    parser.add_argument(
        "--make_fasta",
        action="store_true"
    )
    
    parser.add_argument(
        "--bulk_compute",
        action="store_true"
    )

    parser.add_argument(
        "--concat_tensors",
        action="store_true"
    )

    parser.add_argument(
        "--truncate",
        action="store_true"
    )


    return parser

def main(args):

    split = split_dict[args.split]
    dataset = re.findall(r'(\w*)\_', args.split)[0]
    train, val, test, max_length = load_dataset(dataset, split+'.csv') 

    test_len = len(test)
    val_len = len(val)
    train_len = len(train)
    max_length += 2

    PATH = Path.cwd() / dataset / args.esm_version / split 
    PATH.mkdir(parents=True, exist_ok=True)

    if args.make_fasta:

        print('making fasta files')

        with open(PATH / 'train.fasta', 'w') as train_file:
            writer = fastaparser.Writer(train_file)
            for i, seq in enumerate(train.sequence):
                writer.writefasta((str(i), seq))
        
        with open(PATH / 'test.fasta', 'w') as test_file:
            writer = fastaparser.Writer(test_file)
            for i, seq in enumerate(test.sequence):
                writer.writefasta((str(i), seq))

        with open(PATH / 'val.fasta', 'w') as val_file:
            writer = fastaparser.Writer(val_file)
            for i, seq in enumerate(val.sequence):
                writer.writefasta((str(i), seq))
    
    if args.bulk_compute:
        print('sending command line arguments')

        if args.truncate:
            exit_code = os.system('python esm/scripts/extract.py ' + esm_dict[args.esm_version] + ' ' + str(PATH / 'train.fasta') + ' ' + str(PATH / 'train') + ' ' + '--repr_layers 33 --include mean per_tok --truncate')
            os.system('python esm/scripts/extract.py ' + esm_dict[args.esm_version] + ' ' + str(PATH / 'test.fasta') + ' ' + str(PATH / 'test') + ' ' + '--repr_layers 33 --include mean per_tok --truncate')
            os.system('python esm/scripts/extract.py ' + esm_dict[args.esm_version] + ' ' + str(PATH / 'val.fasta') + ' ' + str(PATH / 'val') + ' ' + '--repr_layers 33 --include mean per_tok --truncate')

        else:
            exit_code = os.system('python esm/scripts/extract.py ' + esm_dict[args.esm_version] + ' ' + str(PATH / 'train.fasta') + ' ' + str(PATH / 'train') + ' ' + '--repr_layers 33 --include mean per_tok')
            os.system('python esm/scripts/extract.py ' + esm_dict[args.esm_version] + ' ' + str(PATH / 'test.fasta') + ' ' + str(PATH / 'test') + ' ' + '--repr_layers 33 --include mean per_tok')
            os.system('python esm/scripts/extract.py ' + esm_dict[args.esm_version] + ' ' + str(PATH / 'val.fasta') + ' ' + str(PATH / 'val') + ' ' + '--repr_layers 33 --include mean per_tok')
        
        if exit_code != 0:
            raise FileNotFoundError("Have you run `git submodule update --init` to populate the `esm` submodule?")

    if args.concat_tensors:
        print('making empty tensors for train set')
        # train set
        if args.truncate:
            embs_aa = torch.zeros([train_len, 1022, 1280])
        else:
            embs_aa = torch.zeros([train_len, max_length, 1280])
        embs_mean = torch.empty([train_len, 1280])
        labels = torch.empty([train_len])

        print('starting tensor concatenation')
        i = 0
        for l in tqdm(train.target):
            e = torch.load(PATH / 'train' / (str(i)+'.pt'))
            aa = e['representations'][33]
            embs_aa[i, :aa.shape[0], :] = aa
            embs_mean[i] = e['mean_representations'][33]
            labels[i] = l
            i += 1

        torch.save(embs_aa, PATH / 'train_aa.pt')
        torch.save(embs_mean, PATH / 'train_mean.pt')
        torch.save(labels, PATH / 'train_labels.pt')

        print('making empty tensors for val set')
        # train set
        if args.truncate:
            embs_aa = torch.zeros([val_len, 1022, 1280])
        else:
            embs_aa = torch.zeros([val_len, max_length, 1280])
        embs_mean = torch.empty([val_len, 1280])
        labels = torch.empty([val_len])

        print('starting tensor concatenation')
        i = 0
        for l in tqdm(val.target):
            e = torch.load(PATH / 'val' / (str(i)+'.pt'))
            aa = e['representations'][33]
            embs_aa[i, :aa.shape[0], :] = aa
            embs_mean[i] = e['mean_representations'][33]
            labels[i] = l
            i += 1

        torch.save(embs_aa, PATH / 'val_aa.pt')
        torch.save(embs_mean, PATH / 'val_mean.pt')
        torch.save(labels, PATH / 'val_labels.pt')
        
        print('making empty tensors for test set')
        # test set
        if args.truncate:
            embs_aa = torch.zeros([test_len, 1022, 1280])
        else:
            embs_aa = torch.zeros([test_len, max_length, 1280])
        embs_mean = torch.empty([test_len, 1280])
        labels = torch.empty([test_len])

        print('starting tensor concatenation')
        i = 0
        for l in tqdm(test.target):
            e = torch.load(PATH /'test'/ (str(i)+'.pt'))
            aa = e['representations'][33]
            embs_aa[i, :aa.shape[0], :] = aa
            embs_mean[i] = e['mean_representations'][33]
            labels[i] = l
            i += 1
            
        torch.save(embs_aa, PATH / 'test_aa.pt')
        torch.save(embs_mean, PATH / 'test_mean.pt')
        torch.save(labels, PATH / 'test_labels.pt')

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
