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

import argparse 

sys.path.append('.../FLIP/baselines/')
from utils import load_dataset

split_dict = {
    'aav_dt1': 'design_task_regression' ,
    'aav_dt2': 'design_task_reversed_regression',
    'aav_nt1': 'natural_task_1_regression',
    'aav_nt2': 'natural_task_2_regression',
    'cas_neg': 'pi_domain_log_negative_selection_regression',
    'cas_pos': 'pi_domain_log_positive_selection_regression',
    'meltome_clust' : 'clustered_task',
    'meltome_full' : 'full_task',
    'meltome_mixed' : 'mixed_task',
    'gb1_1': 'four_mutations_task_1',
    'gb1_2': 'four_mutations_task_2',
    'gb1_3': 'four_mutations_task_3',
    'gb1_4': 'four_mutations_task_4'
}

esm_dict = {
    'esm1v': 'esm1v_t33_650M_UR90S_1', 
    'esm1b': 'esm1b_t33_650M_UR50S',
    'esm_rand': '/data/v-jodymou/embeddings/random_init_esm1b.pt'
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

def truncate(s):
    with open('/home/v-jodymou/bio-benchmarks/tasks/cas/cas9_sequence.fasta', 'r') as fasta_file:
        parser = fastaparser.Reader(fasta_file)
        for seq in parser:
            cas9_seq = seq.sequence_as_string()

    if len(s) <= 1022:
        return s 
    else:
        if cas9_seq[:1022] in s:
            return s[-1022:]
        else:
            return s[:1022]

def main(args):

    split = split_dict[args.split]
    dataset = re.findall(r'(\w*)\_', args.split)[0]
    train, test, max_length = load_dataset(dataset, split+'.csv', mut_only = False, val_split = False) 

    if dataset == 'cas':
        train.sequence = train.sequence.apply(truncate)
        test.sequence = test.sequence.apply(truncate)

    test_len = len(test)
    train_len = len(train)
    max_length += 2


    PATH = '../FLIP/embeddings/'+dataset+'/'+args.esm_version+'/'+split+'/'

    if args.make_fasta:

        print('making fasta files')

        with open(PATH+'train.fasta', 'w') as train_file:
            writer = fastaparser.Writer(train_file)
            for i, seq in enumerate(train.sequence):
                writer.writefasta((str(i), seq))
        
        with open(PATH+'test.fasta', 'w') as test_file:
            writer = fastaparser.Writer(test_file)
            for i, seq in enumerate(test.sequence):
                writer.writefasta((str(i), seq))
    
    if args.bulk_compute:
        print('sending command line arguments')

        if args.truncate:
            os.system('python /data/v-jodymou/embeddings/esm/extract.py ' + esm_dict[args.esm_version] + ' ' + PATH + 'train.fasta ' + PATH + 'train/ --repr_layers 33 --include mean per_tok --truncate --gpu '+args.gpu)
            os.system('python /data/v-jodymou/embeddings/esm/extract.py ' + esm_dict[args.esm_version] + ' ' + PATH + 'test.fasta ' + PATH + 'test/ --repr_layers 33 --include mean per_tok --truncate --gpu '+args.gpu)
        else:
            os.system('python /data/v-jodymou/embeddings/esm/extract.py ' + esm_dict[args.esm_version] + ' ' + PATH + 'train.fasta ' + PATH + 'train/ --repr_layers 33 --include mean per_tok --gpu '+args.gpu)
            os.system('python /data/v-jodymou/embeddings/esm/extract.py ' + esm_dict[args.esm_version] + ' ' + PATH + 'test.fasta ' + PATH + 'test/ --repr_layers 33 --include mean per_tok --gpu '+args.gpu)
    
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
            e = torch.load(PATH+'train/'+str(i)+'.pt')
            aa = e['representations'][33]
            embs_aa[i, :aa.shape[0], :] = aa
            embs_mean[i] = e['mean_representations'][33]
            labels[i] = l
            i += 1

        torch.save(embs_aa, PATH+'train_aa.pt')
        torch.save(embs_mean, PATH+'train_mean.pt')
        torch.save(labels, PATH+'train_labels.pt')
        
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
            e = torch.load(PATH+'test/'+str(i)+'.pt')
            aa = e['representations'][33]
            embs_aa[i, :aa.shape[0], :] = aa
            embs_mean[i] = e['mean_representations'][33]
            labels[i] = l
            i += 1
            
        torch.save(embs_aa, PATH+'test_aa.pt')
        torch.save(embs_mean, PATH+'test_mean.pt')
        torch.save(labels, PATH+'test_labels.pt')

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)