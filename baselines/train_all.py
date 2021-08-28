import numpy as np
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import re 
from csv import writer

sys.path.append('../FLIP/baselines/')

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
    'meltome_mixed' : 'mixed_task',
    'gb1_1': 'four_mutations_task_1',
    'gb1_2': 'four_mutations_task_2',
    'gb1_3': 'four_mutations_task_3',
    'gb1_4': 'four_mutations_task_4'
}


def create_parser():
    parser = argparse.ArgumentParser(
        description="train esm"
    )

    parser.add_argument(
        "split",
        choices = ["aav_dt1", "aav_dt2", "aav_nt1", "aav_nt2", "cas_neg", "cas_pos", "meltome_clust", "meltome_full", "meltome_mixed", "gb1_1", "gb1_2", "gb1_3", "gb1_4"],
        type=str,
    )

    parser.add_argument(
        "model",
        choices = ["ridge", "levenshtein", "cnn", "esm1b", "esm1v", "esm_rand"],
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

    parser.add_argument(
        "--mut_mean", 
        action="store_true"
    )

    parser.add_argument(
        "--random_sample", 
        action="store_true"
    )

    parser.add_argument(
        "--flip", 
        action="store_true"
    )

    return parser

def train_eval(dataset, model, split, device, mean, mut_mean, samples, index, batch_size, flip): 
    # could get utils to output iterators directly, input batch size?
    if model.startswith('esm'): # if training an esm model:
        train_data, val_data, test_data, max_length = load_esm_dataset(dataset, model, split, mean, mut_mean, samples, index, flip)
        # 560, 590
    
    else:
        pass # load data normally

    train_iterator = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_iterator = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_iterator = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    EVAL_PATH = Path.cwd() / 'evals' / dataset / model / split 

    if model.startswith('esm'): # train and evaluate the ESM
        if mean or mut_mean: 
            esm_linear = ESMAttention1dMean(d_embedding=1280)
            mean = True
        else:
            esm_linear = ESMAttention1d(max_length=max_length, d_embedding=1280)   
        optimizer = optim.Adam(esm_linear.parameters(), lr=0.001)
        criterion = nn.MSELoss() 

        epochs_trained = train_esm(train_iterator, val_iterator, esm_linear, device, criterion, optimizer, 100, mean)
        
        train_rho, train_mse = evaluate_esm(train_iterator, esm_linear, device, len(train_data), mean, mut_mean, EVAL_PATH / 'test')
        test_rho, test_mse = evaluate_esm(test_iterator, esm_linear, device, len(test_data), mean, mut_mean, EVAL_PATH / 'train')
        
        # TODO: append to a CSV here
        

    if model == 'ridge':
        pass

    if model == 'levenshtein':
        pass

    if model == 'cnn':
        pass
    
    print('train stats: Spearman: %.2f MSE: %.2f ' % (train_rho, train_mse))
    print('test stats: Spearman: %.2f MSE: %.2f ' % (test_rho, test_mse))

    with open(Path.cwd() / 'evals'/ 'results.csv', 'a', newline='') as f:
        if args.mean:
            model+='_mean'
        if args.mut_mean:
            model+='_mut_mean'
        if args.flip:
            split+='_flipped'
        writer(f).writerow([dataset, model, split, index, train_rho, train_mse, test_rho, test_mse, epochs_trained])


def main(args):

    torch.manual_seed(10)
    random.seed(10)

    device = torch.device('cuda:'+args.gpu)
    split = split_dict[args.split]
    dataset = re.findall(r'(\w*)\_', args.split)[0]

    print('dataset: {0} model: {1} split: {2} \n'.format(dataset, args.model, split))


    if args.random_sample:   
        # make a dictionary mapping the split to the indices
        train, _, _ = load_dataset(dataset, split+'.csv', mut_only=False, val_split=False) 
        samples = {}
        for i in range(100):
            train = train.reset_index(drop=True)
            samples[i] = train.index[train['random_sample'] =='S'+str(i+1)].to_numpy()
        
        # then, run training and evaluation on every random sample
        for i in range(100):
            train_eval(dataset, args.model, split, device, args.mean, args.mut_mean, samples=samples, index=i, batch_size=96, flip=args.flip)

    else:
        train_eval(dataset, args.model, split, device, args.mean, args.mut_mean, samples=None, index=None, batch_size=256, flip=args.flip)
    
    
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)