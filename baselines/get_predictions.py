import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np

from esm.models.esmc import ESMC
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

from sequence_models.collaters import Seq2PropertyCollater
from sequence_models.constants import PAD
from sequence_models.structure import Attention1d
from sequence_models.utils import warmup
from sequence_models.flip_utils import load_flip_data
from sequence_models.pretrained import load_model_and_alphabet





class Model(nn.Module):

    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.attention = Attention1d(d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, e, input_mask=None):
        attended = self.attention(e, input_mask=input_mask)
        hidden = self.hidden(self.activation(attended))
        return self.linear(self.dropout(self.activation(hidden)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_fpath', type=str)
    parser.add_argument('out_fpath', type=str)

    args = parser.parse_args()
    train(args)


def train(args):
    _ = torch.manual_seed(0)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')

    np.random.seed(0)
    carp, carp_collater = load_model_and_alphabet('carp_640M')
    embedder = carp.model.embedder.to(device)
    d_model = carp.model.embedder.up_embedder.conv.out_channels
    decoder = Model(d_model, dropout=0)
    decoder = decoder.to(device)
    carp_model = nn.ModuleDict({'embedder': embedder, 'decoder': decoder})
    carp_model = carp_model.eval()
    carp_alphabet = carp_collater.tokenizer.alphabet
    carp_collate_fn = Seq2PropertyCollater(carp_alphabet, return_mask=True)
    esm = ESMC.from_pretrained("esmc_300m") # or "cpu"
    esm = esm.to(device)
    esm_tokenizer = EsmSequenceTokenizer()

    def esm_collator(batch):
        data = tuple(zip(*batch))
        seqs, labels = data
        t = [torch.tensor(esm_tokenizer.encode(s)) for s in seqs]
        max_len = max([len(tt) for tt in t])
        t = [F.pad(tt, (0, max_len - len(tt)), value=1) for tt in t]
        t = torch.stack(t)
        y = torch.tensor(labels).unsqueeze(-1).float()
        input_mask = t != 1
        return t, y, input_mask
    d_model = 960
    decoder = Model(d_model, dropout=0).to(device)
    esm_model = nn.ModuleDict({'embed': esm.embed, 'transformer': esm.transformer, 'decoder': decoder})
    esm_model = esm_model.eval()

    split_dict = {
        "AMY_BACSU": {
            "random.csv": "random_split.csv",
            "by_position.csv": "hard_split_.csv",
            "close_to_far.csv":"med_split_is_close_to_as_0.csv",
            "far_to_close.csv": "med_split_is_close_to_as_1.csv",
            "one_to_many.csv": "one_to_many.csv",
        },
        "hydro": {
            "low_to_high.csv": "hard_split.csv",
            "random.csv": "random_split.csv",
            "three_to_many.csv": "easy_split.csv",
            "to_06241.csv":   "med_P06241test_split.csv",
            "to_P01053.csv": "med_P01053test_split.csv",
            "to_P0A9X9.csv": "med_P0A9X9test_split.csv"
        },
        "ired": {
            "mutation_order.csv": "ired_mutation_order_split.csv",
            "random.csv": "ired_random_split.csv",
        },
        "NucB": {
            "random.csv": "easy.csv",
            "two_to_many.csv": "two_to_many.csv",
        },
        "PDZ3": {
            "random.csv": "rand_split.csv",
            "single_to_double.csv": "single_to_double.csv",
        },
        "RhoMax": {
            "by_wt.csv": "by_wt.csv",
        },
        "trpb": {
            "by_position.csv": "trpB_no_position_overlap_split.csv",
            "one_to_many.csv": "trpB_one_vs_many_split.csv",
            "two_to_many.csv": "trpB_two_vs_many_split.csv"
        }
    }
    ## Grab data
    checkpoint_paths = ["/mnt/amlt/flip_results_5-9-2025", "/mnt/amlt/flip_results/"]
    os.makedirs(args.out_fpath, exist_ok=True)
    landscapes = os.listdir(args.data_fpath)
    for landscape in landscapes:
        split_csvs = os.listdir(os.path.join(args.data_fpath, landscape, "splits"))
        split_csvs = [csv for csv in split_csvs if ".csv" in csv]
        num_workers = 4
        for split_csv in split_csvs[::-1]:
            out_file = os.path.join(args.out_fpath, "%s_%s_predictions.csv" % (landscape, split_csv[:-4]))
            if os.path.isfile(out_file):
                continue
            print(landscape, split_csv, datetime.now())
            ds_train, ds_valid, ds_test = load_flip_data(args.data_fpath, landscape, split_csv[:-4], max_len=2048,
                                                         scale=True)
            results_df = pd.DataFrame()
            results_df['sequence'] = [d[0] for d in ds_test]
            results_df['scaled_target'] = [d[1] for d in ds_test]

            carp_dl_test = DataLoader(ds_test, batch_size=32, collate_fn=carp_collate_fn, num_workers=num_workers)
            esm_dl_test = DataLoader(ds_test, batch_size=32, collate_fn=esm_collator, num_workers=num_workers)
            task = landscape + "_" + split_dict[landscape][split_csv][:-4]
            for seed in range(5):
                for weights in ['pretrained', 'naive']:
                    checkpoint_stem = 'carp_%s_%s_%d' % (task, weights, seed)
                    try:
                        sd = torch.load(os.path.join(checkpoint_paths[0], checkpoint_stem + "_best.pt"),
                                        weights_only=False)
                    except FileNotFoundError:
                        sd = torch.load(os.path.join(checkpoint_paths[1], checkpoint_stem + "_best.pt"),
                                        weights_only=False)
                    carp_model.load_state_dict(sd['model_state_dict'])
                    predictions = []
                    for i, batch in enumerate(carp_dl_test):
                        src, tgt, input_mask = batch
                        src = src.to(device)
                        with torch.no_grad():
                            input_mask = (src != carp_alphabet.index(PAD)).float().unsqueeze(-1)
                            e = carp_model['embedder'](src, input_mask=input_mask)
                            predictions.append(carp_model['decoder'](e, input_mask=input_mask).detach().cpu().numpy())
                    predictions = np.concatenate(predictions)
                    results_df["carp_%s_%d" % (weights, seed)] = predictions

                    checkpoint_stem = 'esmc_%s_%s_%d' % (task, weights, seed)
                    try:
                        sd = torch.load(os.path.join(checkpoint_paths[0], checkpoint_stem + "_best.pt"),
                                        weights_only=False)
                    except FileNotFoundError:
                        sd = torch.load(os.path.join(checkpoint_paths[1], checkpoint_stem + "_best.pt"),
                                        weights_only=False)
                    esm_model.load_state_dict(sd['model_state_dict'])
                    predictions = []
                    for i, batch in enumerate(esm_dl_test):
                        src, tgt, input_mask = batch
                        src = src.to(device)
                        input_mask = input_mask.to(device)
                        with torch.no_grad():
                            e = esm_model['embed'](src)
                            e = esm_model['transformer'](e)[0].float()
                            predictions.append(esm_model['decoder'](e, input_mask=input_mask).detach().cpu().numpy())
                    predictions = np.concatenate(predictions)
                    results_df["esmc_%s_%d" % (weights, seed)] = predictions
            results_df.to_csv(out_file,
                              index=False)


if __name__ == '__main__':
    main()