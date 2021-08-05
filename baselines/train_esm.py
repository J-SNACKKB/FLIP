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

sys.path.append('/home/v-jodymou/bio-benchmarks/baselines/')

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
        description="train the base of linear model"
    )

    parser.add_argument(
        "split",
        type=str,
    )

    parser.add_argument(
        "gpu",
        type=str
    )

    return parser

def main(args):
    
    device = torch.device('cuda:'+args.gpu)
    split = split_dict[args.split]
    dataset = re.findall(r'(\w*)\_', args.split)[0]

    batch_size = 256

    PATH = '/data/v-jodymou/embeddings/'+dataset+'/1b/'+split+'/'

    #train = torch.load(PATH + 'train_aa.pt') #data_len x seq x 1280
    #train_l = torch.load(PATH + 'train_labels.pt') 

    test = torch.load(PATH + 'test_aa.pt') #data_len x seq x 1280
    test_l = torch.load(PATH + 'test_labels.pt')

    # TEMPORARY FIX TODO: resave without the zeros!!!
    test = test[:test_l.shape[0], :, :]

    print(test.shape)
    print(test_l.shape) 

    print('data loaded')

    idx = random.sample(range(0, train.shape[0]), train.shape[0]//10) 
    idx_r = [i for i in np.arange(train.shape[0]) if i not in idx]

    train_esm_data = TensorDataset(train[idx_r], train_l[idx_r])
    val_esm_data = TensorDataset(train[idx], train_l[idx])
    test_esm_data = TensorDataset(test, test_l)

    train_esm_iterator = DataLoader(train_esm_data, batch_size=batch_size, shuffle=True)
    val_esm_iterator = DataLoader(val_esm_data, batch_size=batch_size, shuffle=True)
    test_esm_iterator = DataLoader(test_esm_data, batch_size=batch_size, shuffle=True)

    esm_linear = ESMAttention1d(max_length=train.shape[1], d_embedding=1280) 
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(esm_linear.parameters(), lr=0.001)

    train_esm_linear(train_esm_iterator, val_esm_iterator, device, esm_linear, criterion, optimizer, 100)
        
    print('starting evaluation')

    EVAL_PATH = 'evals/'+dataset+'/esm/'+split+'/esm1b/'

    evaluate_esm(train_esm_iterator, device, esm_linear, len(train_esm_data), EVAL_PATH)
    evaluate_esm(test_esm_iterator, device, esm_linear, len(test_esm_data), EVAL_PATH)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)