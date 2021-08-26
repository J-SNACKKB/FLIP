import numpy as np
import time
import random
import sys

import os
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist 
from torch.utils.data import TensorDataset, DistributedSampler, DataLoader
import pickle 

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns 
import re 

sys.path.append('/home/v-jodymou/FLIP-benchmarks/baselines/')

from utils import *
from evals import *
from models import * 
from train import * 

import argparse 

split_dict = {
    'aav_dt1': 'design_task_regression' ,
    'aav_dt2': 'design_task_reversed_regression',
    'aav_nt1': 'natural_task_1_regression',
    'aav_nt2': 'natural_task_2_regression',
    'cas_neg': 'pi_domain_log_negative_selection_regression',
    'cas_pos': 'pi_domain_log_positive_selection_regression',
    'meltome_clust' : 'clustered_task',
    'meltome_full' : 'full_task',
    'meltome_mixed' : 'mixed_task'
}

def create_parser():
    parser = argparse.ArgumentParser(
        description="train esm"
    )

    parser.add_argument(
        "split",
        choices = ["aav_dt1", "aav_dt2", "aav_nt1", "aav_nt2", "cas_neg", "cas_pos", "meltome_clust", "meltome_full", "meltome_mixed"],
        type=str,
    )

    parser.add_argument(
        "esm_type",
        choices = ["esm1b", "esm1v", "esm1b_rand", "esm1v_rand"],
        type = str
    )

    parser.add_argument(
        "gpu",
        type=str
    )

    parser.add_argument(
        "--mean", 
        action="store_true"
    )

    return parser

def main(args):
    
    torch.manual_seed(10)
    random.seed(10)
    device = torch.device('cuda:'+args.gpu)
    split = split_dict[args.split]
    dataset = re.findall(r'(\w*)\_', args.split)[0]

    batch_size = 256

    PATH = '/data/v-jodymou/embeddings/' + dataset + '/' + args.esm_type + '/' + split + '/'

    if args.mean:
        train = torch.load(PATH + 'train_mean.pt') #data_len x seq x 1280
        test = torch.load(PATH + 'test_mean.pt') #data_len x seq x 1280
    else:
        train = torch.load(PATH + 'train_aa.pt') #data_len x seq x 1280
        test = torch.load(PATH + 'test_aa.pt') #data_len x seq x 1280

    train_l = torch.load(PATH + 'train_labels.pt') 
    test_l = torch.load(PATH + 'test_labels.pt')

    # TEMPORARY FIX TODO: resave without the zeros!!!
    test = test[:test_l.shape[0]]


    print('data loaded')

    idx = random.sample(range(0, train.shape[0]), train.shape[0]//10) 
    idx_r = [i for i in np.arange(train.shape[0]) if i not in idx]

    train_esm_data = TensorDataset(train[idx_r], train_l[idx_r])
    val_esm_data = TensorDataset(train[idx], train_l[idx])
    test_esm_data = TensorDataset(test, test_l)

    train_esm_iterator = DataLoader(train_esm_data, batch_size=batch_size, shuffle=True)
    val_esm_iterator = DataLoader(val_esm_data, batch_size=batch_size, shuffle=True)
    test_esm_iterator = DataLoader(test_esm_data, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss() 

    if args.mean: # mean embeddings
        esm_linear = ESMAttention1dMean(d_embedding=1280)
        optimizer = optim.Adam(esm_linear.parameters(), lr=0.001)
        train_esm_linear_mean(train_esm_iterator, val_esm_iterator, device, esm_linear, criterion, optimizer, 100)
        print('starting evaluation for', args.esm_type, dataset, split, 'mean pool embedding')

    else: # per AA embeddings 
        esm_linear = ESMAttention1d(max_length=train.shape[1], d_embedding=1280) 
        optimizer = optim.Adam(esm_linear.parameters(), lr=0.001) 
        train_esm_linear(train_esm_iterator, val_esm_iterator, device, esm_linear, criterion, optimizer, 100)
        print('starting evaluation for', args.esm_type, dataset, split, 'per AA embedding')
    

    EVAL_PATH = 'evals/' + dataset + '/' + args.esm_type + '/' + split

    print('evaluating train:')
    if args.mean:
        evaluate_esm_mean(train_esm_iterator, device, esm_linear, len(train_esm_data), EVAL_PATH+'_train_mean')
    else:
        evaluate_esm(train_esm_iterator, device, esm_linear, len(train_esm_data), EVAL_PATH+'_train')

    print('evaluating test:')
    if args.mean:
        evaluate_esm_mean(test_esm_iterator, device, esm_linear, len(test_esm_data), EVAL_PATH+'_test_mean')
    else:
        evaluate_esm(test_esm_iterator, device, esm_linear, len(test_esm_data), EVAL_PATH+'_test')


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)