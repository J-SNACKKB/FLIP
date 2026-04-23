import argparse
import json
import os
from tqdm import tqdm

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score, roc_auc_score

import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd

torch.manual_seed(0)

from sequence_models.utils import Tokenizer
from sequence_models.flip_utils import load_flip_data

AAINDEX_ALPHABET = 'ARNDCQEGHILKMFPSTWYV'


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--solver', type=str, default='auto')
parser.add_argument('--max_iter', type=int, default=1000000)
parser.add_argument('--tol', type=float, default=1e-5)
args = parser.parse_args()

results_path = "/home/kevyan/results/flipv3/"

model_dict = {
    "esm2_650M_scores": "esm2_650M_zs",
    # "dayhoff_fwd": "dayhoff_3b_gr_hm_c_fwd_zs",
    # "dayhoff_bwd": "dayhoff_3b_gr_hm_c_bwd_zs",
    # "dayhoff_min": "dayhoff_3b_gr_hm_c_min_zs",
    # "dayhoff_max": "dayhoff_3b_gr_hm_c_max_zs",
    "dayhoff_mean": "dayhoff_3b_gr_hm_c_mean_zs",
    # "dayhoff_3bur90_fwd": "dayhoff_3b_ur90_fwd_zs",
    # "dayhoff_3bur90_bwd": "dayhoff_3b_ur90_bwd_zs",
    # "dayhoff_3bur90_min": "dayhoff_3b_ur90_zs",
    # "dayhoff_3bur90_max": "dayhoff_3b_ur90_max_zs",
    "dayhoff_3bur90_mean": "dayhoff_3b_ur90_mean_zs",
    "dayhoff_both_mean": "dayhoff_both_mean"
}
tokenizer = Tokenizer(AAINDEX_ALPHABET) # tokenize

# Randomize at different data sizes
input_path = "/home/kevyan/results/flipv3/zs/"
landscapes = os.listdir(input_path)
np.random.seed(23)
replicates = 50
results = pd.DataFrame(columns=['dataset', 'model', 'fraction_train', 'n_train', 'Spearman', 'replicate'])
n_min = 50
with tqdm(total=len(landscapes)  * replicates) as pbar:
    for landscape in landscapes:
        df = pd.read_csv(os.path.join(input_path, landscape))
        landscape_name = landscape[:-7]
        n = len(df)
        X = df['sequence'].values
        X = [torch.tensor(tokenizer.tokenize(i.replace(":", ""))).view(-1, 1) for i in X]
        maxlen = max([len(i) for i in X])
        X = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]), "constant", 0.) for i in X]
        X_enc = []  # ohe
        for i in X:
            i_onehot = torch.FloatTensor(maxlen, len(AAINDEX_ALPHABET))
            i_onehot.zero_()
            i_onehot.scatter_(1, i, 1)
            X_enc.append(i_onehot)
        X_enc = np.array([np.array(i.view(-1)) for i in X_enc])  # flatten
        cols = ['esm2_650M_scores', 'carp_640m_zs', 'dayhoff']
        X_zs = df[cols].values
        new_X = np.hstack([X_enc, X_zs])
        log10_min = np.log10(n_min)
        log10_max = np.log10(n * 0.8)
        n_trains = np.logspace(log10_min, log10_max, num=10)
        for rep in range(replicates):
            n_train = int(n_trains[rep % 10])
            fraction = n_train / n
            n_test = n - n_train
            idx = np.arange(n)
            np.random.shuffle(idx)
            y_scale = df.iloc[idx[:n_train]]['target'].values[:, None]
            y_test = df.iloc[idx[n_train:]]['target'].values[:, None]
            X_train_enc = X_enc[idx[:n_train]]
            X_test_enc = X_enc[idx[n_train:]]
            scaler = StandardScaler()
            scaler.fit(y_scale)
            y_train = scaler.transform(y_scale)
            y_test = scaler.transform(y_test)
            lr = Ridge(solver='auto', tol=1e-5, max_iter=1000000, alpha=10)
            # lr = Ridge(solver=args.solver, tol=args.tol, max_iter=args.max_iter, alpha=10)
            lr.fit(X_train_enc, y_train)
            preds = lr.predict(X_test_enc)
            preds = preds.reshape(-1, 1)
            rho = spearmanr(y_test, preds).correlation
            results.loc[len(results)] = [landscape_name, 'Ridge (one-hot)', fraction, n_train, rho, rep]
            new_X_train = new_X[idx[:n_train]]
            new_X_test = new_X[idx[n_train:]]
            lr = Ridge(solver='auto', tol=1e-5, max_iter=1000000, alpha=10)
            lr.fit(new_X_train, y_train)
            preds = lr.predict(new_X_test)
            rho = spearmanr(y_test, preds).correlation
            results.loc[len(results)] = [landscape_name, 'Ridge (one-hot + likelihoods)', fraction, n_train, rho, rep]
            pbar.update(1)
results.to_csv(results_path + "random_ridge.csv", index=False)

input_path = "/home/kevyan/data/flip_data_pruned/"
landscapes = os.listdir(input_path)

for landscape in landscapes:
    split_csvs = os.listdir(os.path.join(input_path, landscape, "splits"))
    split_csvs = [c for c in split_csvs if "csv" in c]
    for split_csv in split_csvs:
        df = pd.read_csv(os.path.join(input_path, landscape, "splits", split_csv))
# tokenize train data
        X_train = df[(df["set"] == "train") & (~df['validation'])]['sequence'].values
        y_scale = df[df['set'] == "train"]['target'].values[:, None]
        y_train = df[(df["set"] == "train") & (~df['validation'])]['target'].values[:, None]
        X_train = [torch.tensor(tokenizer.tokenize(i.replace(":", ""))).view(-1, 1) for i in X_train]
        seq_test = df[df["set"] == "test"]['sequence'].values
        y_test = df[df["set"] == "test"]['target'].values[:, None]
        X_test = [torch.tensor(tokenizer.tokenize(i.replace(":", ""))).view(-1, 1) for i in seq_test]
        # padding
        maxlen_train = max([len(i) for i in X_train])
        maxlen_test = max([len(i) for i in X_test])
        maxlen = max([maxlen_train, maxlen_test])

        X_train = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]), "constant", 0.) for i in X_train]
        X_train_enc = [] # ohe
        for i in X_train:
            i_onehot = torch.FloatTensor(maxlen, len(AAINDEX_ALPHABET))
            i_onehot.zero_()
            i_onehot.scatter_(1, i, 1)
            X_train_enc.append(i_onehot)
        X_train_enc = np.array([np.array(i.view(-1)) for i in X_train_enc]) # flatten

        X_test = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]), "constant", 0.) for i in X_test]
        X_test_enc = [] # ohe
        for i in X_test:
            i_onehot = torch.FloatTensor(maxlen, len(AAINDEX_ALPHABET))
            i_onehot.zero_()
            i_onehot.scatter_(1, i, 1)
            X_test_enc.append(i_onehot)
        X_test_enc = np.array([np.array(i.view(-1)) for i in X_test_enc]) # flatten
        scaler = StandardScaler()
        scaler.fit(y_scale)
        y_train = scaler.transform(y_train)
        y_test = scaler.transform(y_test)
        lr = Ridge(solver=args.solver, tol=args.tol, max_iter=args.max_iter, alpha=10)
        lr.fit(X_train_enc, y_train)

        print(landscape, split_csv[:-4], 'one-hot')
        preds = lr.predict(X_test_enc)
        preds = preds.reshape(-1, 1)

        mse = mean_squared_error(y_test, preds)
        print('TEST MSE: ', mse)
        print('TEST RHO: ', spearmanr(y_test, preds).correlation)
        y_test_pos = y_test - y_test.min()
        print('TEST NDCG: ', ndcg_score(y_test_pos.T, preds.T))

        results = pd.DataFrame()
        results['sequence'] = seq_test
        results['target'] = y_test
        results['prediction'] = preds
        os.makedirs(os.path.join(results_path, "ridge", landscape), exist_ok=True)
        results.to_csv(os.path.join(results_path, "ridge", landscape, split_csv), index=False)


        if landscape != "PDZ3":
            df['dayhoff'] = df['dayhoff_bwd'] + df['dayhoff_fwd'] + df['dayhoff_3bur90_fwd'] + df['dayhoff_3bur90_bwd']
            if 'carp_640m_masked_zs' in df.columns:
                cols = ['dayhoff', "esm2_650M_scores", 'carp_640m_masked_zs']
            else:
                cols = ['dayhoff', "esm2_650M_scores", 'carp_640m_zs']
        else:
            df['dayhoff1'] = df['dayhoff_bwd_1'] + df['dayhoff_fwd_1'] + df['dayhoff_3bur90_fwd_1'] + df['dayhoff_3bur90_bwd_1']
            df['dayhoff2'] = df['dayhoff_bwd_2'] + df['dayhoff_fwd_2'] + df['dayhoff_3bur90_fwd_2'] + df['dayhoff_3bur90_bwd_2']
            cols = ["dayhoff1", "dayhoff2", "esm2_650M_scores1", "esm2_650M_scores1", 'carp_640m_zs_1', 'carp_640m_zs_2']
        zs_x_train = df[(df["set"] == "train") & (~df['validation'])][cols].values
        new_x_train = np.hstack([X_train_enc, zs_x_train])
        zs_x_test = df[df["set"] == "test"][cols].values
        new_x_test = np.hstack([X_test_enc, zs_x_test])
        lr = Ridge(solver=args.solver, tol=args.tol, max_iter=args.max_iter, alpha=10)

        lr.fit(new_x_train, y_train)

        print(landscape, split_csv[:-4], 'one-hot + zs')
        preds = lr.predict(new_x_test)
        preds = preds.reshape(-1, 1)
        mse = mean_squared_error(y_test, preds)
        print('TEST MSE: ', mse)
        print('TEST RHO: ', spearmanr(y_test, preds).correlation)
        y_test_pos = y_test - y_test.min()
        print('TEST NDCG: ', ndcg_score(y_test_pos.T, preds.T))
        results = pd.DataFrame()
        results['sequence'] = seq_test
        results['target'] = y_test
        results['prediction'] = preds
        os.makedirs(os.path.join(results_path, "ridge_zs", landscape), exist_ok=True)
        results.to_csv(os.path.join(results_path, "ridge_zs", landscape, split_csv), index=False)


