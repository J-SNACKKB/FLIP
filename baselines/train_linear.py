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
    'cas_pos': 'pi_domain_log_positive_selection_regression'
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

    train, val, test, max_length = load_dataset(dataset, split+'.csv', mut_only = False, val_split = True) 

    batch_size = 256

    train_data = get_data(train, max_length, encode_pad = True, zip_dataset = True)
    val_data = get_data(val, max_length, encode_pad = True, zip_dataset = True)
    test_data = get_data(test, max_length, encode_pad = True, zip_dataset = True)

    train_iterator = DataLoader(SequenceDataset(train_data), batch_size = batch_size, shuffle = True) 
    val_iterator = DataLoader(SequenceDataset(val_data), batch_size = batch_size, shuffle = True) 
    test_iterator = DataLoader(SequenceDataset(test_data), batch_size = batch_size, shuffle = True) 

    linear = Linear_Base(max_length) 
    optimizer = optim.Adam(linear.parameters(), lr = 0.001, weight_decay = 0.0) #weight_decay is not exact same as L2 loss bc used in update step?

    train_linear(train_iterator, val_iterator, device, linear, optimizer,  r_b = 'r', epoch_num = 100)
    
    # eval
    EVAL_PATH = 'evals/'+dataset+'/linear/' + split
    
    evaluate_linear(train_iterator, linear, device, 'r', EVAL_PATH+'_train') # creates tuples
    evaluate_linear(test_iterator, linear, device, 'r', EVAL_PATH+'_test') # 


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)