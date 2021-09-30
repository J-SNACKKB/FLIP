import numpy as np
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import re 
from csv import writer

sys.path.append('/.../.../FLIP/baselines/')

from utils import *
from evals import *
from models import * 
from train import * 

import argparse 

split_dict = {
    'aav_1': 'des_mut' ,
    'aav_2': 'mut_des',
    'aav_3': 'one_vs_many',
    'aav_4': 'two_vs_many',
    'aav_5': 'seven_vs_many',
    'aav_6': 'low_vs_high',
    'aav_7': 'sampled',
    'meltome_mixed' : 'mixed_split',
    'meltome_human' : 'human',
    'meltome_human_cell' : 'human_cell',
    'gb1_1': 'one_vs_rest',
    'gb1_2': 'two_vs_rest',
    'gb1_3': 'three_vs_rest',
    'gb1_4': 'sampled',
    'gb1_5': 'low_vs_high'
}

def create_parser():
    parser = argparse.ArgumentParser(description="train esm")
    parser.add_argument("split", type=str)
    parser.add_argument("model", choices = ["ridge", "cnn", "esm1b", "esm1v", "esm_rand"], type = str)
    parser.add_argument("gpu", type=str)
    parser.add_argument("--mean", action="store_true")
    parser.add_argument("--mut_mean", action="store_true")
    #parser.add_argument("--random_sample", action="store_true") # removed for now as low-N splits were not prepared
    #parser.add_argument("--flip", action="store_true") # for flipping mut-des and des-mut
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--lr", type=float, default=0.001)

    return parser

def train_eval(dataset, model, split, device, mean, mut_mean, samples, index, batch_size, flip, lr): 
    # could get utils to output iterators directly, input batch size?
    if model.startswith('esm'): # if training an esm model:
        
        train_data, val_data, test_data, max_length = load_esm_dataset(dataset, model, split, mean, mut_mean, samples, index, flip)
        # 560, 590
    
    else:
        pass # load data normally

    train_iterator = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_iterator = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_iterator = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    EVAL_PATH = Path.cwd() / 'evals_new' / dataset / model / split 
    EVAL_PATH.mkdir(parents=True, exist_ok=True)

    if model.startswith('esm'): # train and evaluate the ESM
        if mean or mut_mean: 
            esm_linear = ESMAttention1dMean(d_embedding=1280)
            mean = True
        else:
            esm_linear = ESMAttention1d(max_length=max_length, d_embedding=1280)   
        optimizer = optim.Adam(esm_linear.parameters(), lr=lr)
        criterion = nn.MSELoss() 

        epochs_trained = train_esm(train_iterator, val_iterator, esm_linear, device, criterion, optimizer, 500, mean)
        
        train_rho, train_mse = evaluate_esm(train_iterator, esm_linear, device, len(train_data), mean, mut_mean, EVAL_PATH / 'test')
        test_rho, test_mse = evaluate_esm(test_iterator, esm_linear, device, len(test_data), mean, mut_mean, EVAL_PATH / 'train')
        
        # TODO: append to a CSV here
        

    if model == 'ridge':
        pass

    if model == 'cnn':
        pass
    
    print('train stats: Spearman: %.2f MSE: %.2f ' % (train_rho, train_mse))
    print('test stats: Spearman: %.2f MSE: %.2f ' % (test_rho, test_mse))

    with open(Path.cwd() / 'evals_new'/ (dataset+'_results.csv'), 'a', newline='') as f:
        if args.mean:
            model+='_mean'
        if args.mut_mean:
            model+='_mut_mean'
        if args.flip:
            split+='_flipped'
        writer(f).writerow([dataset, model, split, index, train_rho, train_mse, test_rho, test_mse, epochs_trained, lr])


def main(args):

    device = torch.device('cuda:'+args.gpu)
    split = split_dict[args.split]
    dataset = re.findall(r'(\w*)\_', args.split)[0]

    print('dataset: {0} model: {1} split: {2} \n'.format(dataset, args.model, split)) 

    if args.ensemble: 
        for i in range(10):
            random.seed(i)
            torch.manual_seed(i)
            # run training and evaluation on 10 different random seeds 
            train_eval(dataset, args.model, split, device, args.mean, args.mut_mean, samples=None, index=None, batch_size=256, flip=args.flip, lr=args.lr)
    

    else: 
        random.seed(10)
        torch.manual_seed(10)

        if args.random_sample:
            train, _, _ = load_dataset(dataset, split+'.csv', val_split=False) 
            samples = {}
            for i in range(100):
                train = train.reset_index(drop=True)
                samples[i] = train.index[train['random_sample'] =='S'+str(i+1)].to_numpy()
        
            # then, run training and evaluation on every random sample
            for i in range(100):
                train_eval(dataset, args.model, split, device, args.mean, args.mut_mean, samples=samples, index=i, batch_size=96, flip=args.flip, lr=args.lr)


        else:
            train_eval(dataset, args.model, split, device, args.mean, args.mut_mean, samples=None, index=None, batch_size=256, flip=args.flip, lr=args.lr)
    
    
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)